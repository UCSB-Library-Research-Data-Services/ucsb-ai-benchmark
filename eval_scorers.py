import json
import subprocess
import tempfile
import logging
import re
from openai import OpenAI

def extract_code_block(text: str) -> str:
    """Extracts code from a markdown code block."""
    match = re.search(r"```(?:python)?(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

def score_programmatic_code(response_text: str) -> dict:
    """Safely executes python code using subprocess and checks for errors."""
    code = extract_code_block(response_text)
    
    # Create a dummy CSV for pandas to load so the script doesn't fail on missing file
    dummy_csv = "temperature,pressure\n15,1000\n25,1010\n"
    
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = f"{tmpdir}/data.csv"
        script_path = f"{tmpdir}/script.py"
        
        with open(csv_path, "w") as f:
            f.write(dummy_csv)
            
        # modify code to point to the tmp csv if they used 'data.csv'
        code = code.replace("'data.csv'", f"'{csv_path}'").replace('"data.csv"', f'"{csv_path}"')
        
        with open(script_path, "w") as f:
            f.write(code)
            
        try:
            result = subprocess.run(
                ["python3", script_path], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            if result.returncode == 0:
                return {"score": 5, "reason": "Execution successful", "output": result.stdout.strip()}
            else:
                return {"score": 1, "reason": f"Execution failed: {result.stderr.strip()}", "output": ""}
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
    """Checks for an exact string match (useful for multiple choice)."""
    # Just look for the expected letter in the first few characters to be generous
    if expected_match.lower() in response_text.strip()[:10].lower():
         return {"score": 5, "reason": f"Matched exact expected answer: {expected_match}"}
    return {"score": 1, "reason": f"Expected {expected_match}, got {response_text.strip()[:20]}"}

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
