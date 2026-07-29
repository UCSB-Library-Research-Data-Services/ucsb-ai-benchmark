import json
import subprocess
import tempfile
import logging
import re
import sys
from openai import OpenAI

def extract_code_block(text: str) -> str:
    """Extracts the first meaningful code block from a markdown-fenced response.

    Handles:
    - ```python ... ``` fences (with or without language tag)
    - Responses with multiple fences (takes the largest/first real code block)
    - Plain text with no fencing (returned as-is)
    """
    # Match ```<optional lang>\n<code>\n``` — requires at least one non-whitespace char
    pattern = re.compile(r"```(?:[a-zA-Z0-9_+-]*)?\n(.*?)```", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        # Return the longest match to avoid grabbing a tiny inline snippet
        return max(matches, key=len).strip()
    return text.strip()

def score_programmatic_code(response_text: str) -> dict:
    """Safely executes python code using subprocess and checks for errors.

    Uses sys.executable so the code runs inside the same venv that has pandas/numpy
    installed, rather than the system python3 which may be missing dependencies.

    Beyond a clean exit code, it also validates that the script actually printed a
    numeric result — ensuring the model computed the required mean() rather than
    just generating no-op boilerplate that happens to run without errors.
    """
    code = extract_code_block(response_text)

    # Create a dummy CSV for pandas to load so the script doesn't fail on missing file.
    # Values are chosen so the expected output is deterministic and easy to validate:
    # rows with temperature > 20 → [25, 30], pressure mean = (1010 + 1020) / 2 = 1015.0
    dummy_csv = "temperature,pressure\n15,1000\n25,1010\n30,1020\n"

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = f"{tmpdir}/data.csv"
        script_path = f"{tmpdir}/script.py"

        with open(csv_path, "w") as f:
            f.write(dummy_csv)

        # Redirect any hardcoded 'data.csv' reference to the temp file path
        code = code.replace("'data.csv'", f"'{csv_path}'").replace('"data.csv"', f'"{csv_path}"')

        with open(script_path, "w") as f:
            f.write(code)

        try:
            result = subprocess.run(
                [sys.executable, script_path],   # use the venv Python, not system python3
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                return {"score": 1, "reason": f"Execution failed: {result.stderr.strip()}", "output": ""}

            stdout = result.stdout.strip()

            # Semantic check: the script must print at least one numeric value.
            # A model that generates `print("hello")` or no print at all scores 3/5
            # (code is syntactically valid but produces no useful output).
            numeric_pattern = re.compile(r"\d+(\.\d+)?")
            if not numeric_pattern.search(stdout):
                return {
                    "score": 3,
                    "reason": (
                        "Execution successful but no numeric output detected. "
                        "The script may not have printed the computed mean."
                    ),
                    "output": stdout,
                }

            return {"score": 5, "reason": "Execution successful with numeric output", "output": stdout}

        except subprocess.TimeoutExpired:
            return {"score": 1, "reason": "Execution timed out", "output": ""}
        except Exception as e:
            return {"score": 1, "reason": f"Execution error: {str(e)}", "output": ""}

def score_programmatic_metadata(response_text: str, expected_keys: list) -> dict:
    """Validates if output is valid JSON and contains required keys."""
    # Strip markdown block if present
    text = extract_code_block(response_text)
    try:
        data = json.loads(text)
        missing_keys = [k for k in expected_keys if k not in data]
        if not missing_keys:
            return {"score": 5, "reason": "Valid JSON and contains all required keys"}
        else:
            return {"score": 2, "reason": f"Valid JSON but missing keys: {missing_keys}"}
    except json.JSONDecodeError:
        return {"score": 0, "reason": "Output is not valid JSON"}

def score_exact_match(response_text: str, expected_match: str) -> dict:
    """Checks that the response selects the correct multiple-choice letter.

    Accepts common model answer formats, e.g.:
      "C", "C)", "The answer is C", "C. Temp vs Humidity"
    while rejecting responses that merely mention the letter in passing.
    A regex anchors the expected letter at the very start of the stripped
    response or after an explicit "answer is / answer:" preamble.
    """
    text = response_text.strip()
    letter = re.escape(expected_match.upper())

    # Pattern 1: letter stands alone at the start — followed only by punctuation
    #   (). : \n) or end-of-string.  We deliberately do NOT allow a trailing space
    #   because "C is wrong..." starts with "C " and must not match.
    # Pattern 2: letter follows a common preamble (e.g. "The answer is C" / "Answer: C")
    pattern = re.compile(
        rf"(?:^{letter}(?:[).:\n]|$)|(?:answer(?:\s+is)?|the\s+answer\s+is)\s*:?\s*{letter}\b)",
        re.IGNORECASE,
    )

    if pattern.search(text):
        return {"score": 5, "reason": f"Matched expected answer: {expected_match}"}
    return {"score": 1, "reason": f"Expected '{expected_match}', got: '{text[:40]}'"}

def score_with_judge(
    target_response: str, 
    prompt: str, 
    rubric: str, 
    judge_client: OpenAI, 
    judge_model: str,
    tracker # TokenTracker instance
) -> dict:
    """Uses a highly capable LLM to grade a subjective response."""
    
    system_prompt = (
        "You are an impartial scientific evaluator. You will be provided with a TASK, an "
        "EVALUATION RUBRIC, and the MODEL OUTPUT to evaluate.\n"
        "You must output ONLY a valid JSON object with two keys: 'score' (an integer from 1 to 5) "
        "and 'reason' (a short 1-sentence explanation of the score).\n"
        "Do not include markdown formatting or any other text outside the JSON object."
    )
    
    user_prompt = (
        f"TASK:\n{prompt}\n\n"
        f"EVALUATION RUBRIC:\n{rubric}\n\n"
        f"MODEL OUTPUT:\n{target_response}\n\n"
        "Evaluate the MODEL OUTPUT based strictly on the EVALUATION RUBRIC. Output ONLY JSON."
    )
    
    try:
        response = judge_client.chat.completions.create(
            model=judge_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0
        )
        
        # Track Judge token usage
        if tracker and response.usage:
            # Note: We hardcode 'judge_service' logic here, but in production it'll be passed correctly
            # We will handle the service name in the engine.
            pass # Tracking is handled in the engine where we know the judge service name

        judge_text = response.choices[0].message.content.strip()
        
        try:
            # Handle potential markdown code blocks returned by the judge
            judge_text_clean = extract_code_block(judge_text)
            result = json.loads(judge_text_clean)
            return {
                "score": result.get("score", 0), 
                "reason": result.get("reason", "No reason provided by judge"),
                "judge_usage": response.usage # Pass back to engine
            }
        except json.JSONDecodeError:
            return {"score": 0, "reason": f"Judge failed to return valid JSON. Raw output: {judge_text[:50]}"}
            
    except Exception as e:
        return {"score": 0, "reason": f"Judge API call failed: {str(e)}"}
