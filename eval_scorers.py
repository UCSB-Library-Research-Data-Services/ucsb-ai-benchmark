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
    pattern = re.compile(r"```(?:[a-zA-Z0-9_+-]*)?\n(.*?)```", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return max(matches, key=len).strip()
    return text.strip()


# ---------------------------------------------------------------------------
# Numeric output matching helpers (Fix 1)
# ---------------------------------------------------------------------------

def _parse_numbers_from_stdout(stdout: str) -> list[float]:
    """Extract all floating-point/integer numbers from a script's stdout."""
    return [float(m) for m in re.findall(r"-?\d+(?:\.\d+)?", stdout)]


def _numeric_match(stdout: str, expected: list[dict], tolerance: float = 0.01) -> tuple[bool, str]:
    """Check whether expected numeric values appear anywhere in stdout.

    Each entry in `expected` is:
        {"value": <float>, "label": "<human description>"}

    A value matches if any number in stdout is within `tolerance` * abs(value)
    of the expected (relative tolerance), or within 0.01 absolute for near-zero.

    Returns (all_matched: bool, detail_message: str).
    """
    found_numbers = _parse_numbers_from_stdout(stdout)
    missing = []

    for item in expected:
        val = float(item["value"])
        label = item.get("label", str(val))
        abs_tol = max(abs(val) * tolerance, 0.01)
        matched = any(abs(n - val) <= abs_tol for n in found_numbers)
        if not matched:
            missing.append(f"{label} (expected ≈{val}, got {found_numbers})")

    if missing:
        return False, "Missing expected values: " + "; ".join(missing)
    return True, "All expected numeric values found in output"


# ---------------------------------------------------------------------------
# Fix 1: score_programmatic_code
# ---------------------------------------------------------------------------

def score_programmatic_code(
    response_text: str,
    dummy_csv: str = "temperature,pressure\n15,1000\n25,1010\n30,1020\n",
    expected_output: list[dict] | None = None,
    include_output: bool = False,
) -> dict:
    """Safely executes python code using subprocess and validates the output.

    Fix 1 changes vs previous version:
    - Accepts an `expected_output` list from the dataset, e.g.:
          [{"value": 1015.0, "label": "mean pressure for temp>20"}]
    - If provided, checks that the script printed the correct numeric answer
      within a small relative tolerance (±1%).
    - If not provided, falls back to the old "any digit" check (backward-compat).
    - Uses sys.executable (the venv Python) to avoid missing-dependency errors.

    Scoring:
        5/5  – executed cleanly AND all expected values found in output
        3/5  – executed cleanly but numeric output missing/wrong
        1/5  – execution error or timeout
    """
    code = extract_code_block(response_text)

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path  = f"{tmpdir}/data.csv"
        script_path = f"{tmpdir}/script.py"

        with open(csv_path, "w") as f:
            f.write(dummy_csv)

        # Redirect any hardcoded 'data.csv' reference to the temp file path
        patched = code.replace("'data.csv'", f"'{csv_path}'")\
                      .replace('"data.csv"', f'"{csv_path}"')

        with open(script_path, "w") as f:
            f.write(patched)

        try:
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True, text=True, timeout=10
            )

            if result.returncode != 0:
                d = {
                    "score": 1,
                    "reason": f"Execution failed: {result.stderr.strip()}",
                    "output": "",
                }
                if include_output:
                    d["model_output"] = code
                    d["output_type"]  = "code"
                return d

            stdout = result.stdout.strip()

            # ── Semantic check ────────────────────────────────────────────
            if expected_output:
                # Fix 1: verify the correct numeric answer
                matched, detail = _numeric_match(stdout, expected_output)
                if matched:
                    d = {
                        "score":  5,
                        "reason": f"Execution successful and output is correct. {detail}",
                        "output": stdout,
                    }
                else:
                    d = {
                        "score":  3,
                        "reason": f"Execution successful but output is incorrect. {detail}",
                        "output": stdout,
                    }
            else:
                # Backward-compatible fallback: any digit present?
                if re.search(r"\d+(\.\d+)?", stdout):
                    d = {
                        "score":  5,
                        "reason": "Execution successful with numeric output",
                        "output": stdout,
                    }
                else:
                    d = {
                        "score":  3,
                        "reason": (
                            "Execution successful but no numeric output detected. "
                            "The script may not have printed the computed result."
                        ),
                        "output": stdout,
                    }

            if include_output:
                d["model_output"] = code
                d["output_type"]  = "code"
            return d

        except subprocess.TimeoutExpired:
            d = {"score": 1, "reason": "Execution timed out", "output": ""}
            if include_output:
                d["model_output"] = code
                d["output_type"]  = "code"
            return d
        except Exception as e:
            d = {"score": 1, "reason": f"Execution error: {str(e)}", "output": ""}
            if include_output:
                d["model_output"] = code
                d["output_type"]  = "code"
            return d


# ---------------------------------------------------------------------------
# Fix 2: score_programmatic_metadata
# ---------------------------------------------------------------------------

def _normalize_value(v) -> str:
    """Normalize a metadata value for loose comparison.

    - Strip whitespace, lowercase
    - Convert integers to strings (so 2024 ≈ "2024")
    """
    return str(v).strip().lower()


def _values_match(actual_val, expected_val) -> bool:
    """Loose equality: case-insensitive, strip whitespace, int/str interop.

    For strings we also check containment in both directions to handle
    minor wording differences (e.g. "UCSB Dataverse" ⊂ "UCSB Dataverse v4").
    """
    a = _normalize_value(actual_val)
    e = _normalize_value(expected_val)
    return a == e or a in e or e in a


def score_programmatic_metadata(
    response_text: str,
    expected_keys: list,
    expected_values: dict | None = None,
    include_output: bool = False,
) -> dict:
    """Validates JSON output: checks required keys AND (optionally) their values.

    Fix 2 changes vs previous version:
    - Accepts an `expected_values` dict from the dataset, e.g.:
          {"title": "Marine Biology Survey 2023", "publicationYear": 2024}
    - For each key in expected_values, performs a loose value match
      (case-insensitive, whitespace-trimmed, int/str interoperable).
    - Deducts partial score per wrong/missing value.

    Scoring:
        5/5  – valid JSON + all keys present + all expected values correct
        4/5  – valid JSON + all keys present + ≤1 value wrong
        3/5  – valid JSON + all keys present + >1 value wrong
        2/5  – valid JSON + some keys missing (values not checked)
        0/5  – not valid JSON
    """
    text = extract_code_block(response_text)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        d = {"score": 0, "reason": "Output is not valid JSON"}
        if include_output:
            d["model_output"] = text
            d["output_type"]  = "json"
        return d

    # ── Key presence check ────────────────────────────────────────────────
    missing_keys = [k for k in expected_keys if k not in data]
    if missing_keys:
        d = {"score": 2, "reason": f"Valid JSON but missing keys: {missing_keys}"}
        if include_output:
            d["model_output"] = text
            d["output_type"]  = "json"
        return d

    # ── Value correctness check (Fix 2) ──────────────────────────────────
    if expected_values:
        wrong = []
        for key, exp_val in expected_values.items():
            if key not in data:
                wrong.append(f"'{key}' absent")
                continue
            if not _values_match(data[key], exp_val):
                wrong.append(
                    f"'{key}': expected '{exp_val}', got '{data[key]}'"
                )

        if not wrong:
            d = {"score": 5, "reason": "Valid JSON with all required keys and correct values"}
        elif len(wrong) == 1:
            d = {"score": 4, "reason": f"Valid JSON, keys present, but 1 value incorrect: {wrong[0]}"}
        else:
            d = {"score": 3, "reason": f"Valid JSON, keys present, but {len(wrong)} values incorrect: {'; '.join(wrong)}"}
    else:
        # No expected_values provided — old behaviour: keys present = full score
        d = {"score": 5, "reason": "Valid JSON and contains all required keys"}

    if include_output:
        d["model_output"] = text
        d["output_type"]  = "json"
    return d


# ---------------------------------------------------------------------------
# score_exact_match  (unchanged logic, signature kept consistent)
# ---------------------------------------------------------------------------

def score_exact_match(
    response_text: str,
    expected_match: str,
    include_output: bool = False,
) -> dict:
    """Checks that the response selects the correct multiple-choice letter.

    Accepts common model answer formats:
      "C", "C)", "The answer is C", "C. Temp vs Humidity"
    Rejects letters that appear mid-sentence ("C is wrong…").
    """
    text   = response_text.strip()
    letter = re.escape(expected_match.upper())

    pattern = re.compile(
        rf"(?:^{letter}(?:[).:\n]|$)|(?:answer(?:\s+is)?|the\s+answer\s+is)\s*:?\s*{letter}\b)",
        re.IGNORECASE,
    )

    if pattern.search(text):
        d = {"score": 5, "reason": f"Matched expected answer: {expected_match}"}
    else:
        d = {"score": 1, "reason": f"Expected '{expected_match}', got: '{text[:40]}'"}

    if include_output:
        d["model_output"] = text
        d["output_type"]  = "multiple_choice"
    return d


# ---------------------------------------------------------------------------
# Judge helpers
# ---------------------------------------------------------------------------

def extract_judge_feedback(judge_json_response: dict) -> str:
    """Extract multi-line detailed feedback from judge response."""
    return judge_json_response.get(
        "detailed_feedback",
        judge_json_response.get("reason", "")
    )


def score_with_judge(
    target_response: str,
    prompt: str,
    rubric: str,
    judge_client: OpenAI,
    judge_model: str,
    tracker,
    include_detail: bool = False,
) -> dict:
    """Uses a highly capable LLM to grade a subjective response.

    Returns rich detail fields always for model_output/judge_model;
    detailed judge reasoning only when include_detail=True (cost-conscious).
    """

    if include_detail:
        system_prompt = (
            "You are an impartial scientific evaluator. You will be provided with a TASK, an "
            "EVALUATION RUBRIC, and the MODEL OUTPUT to evaluate.\n"
            "You must output ONLY a valid JSON object with three keys:\n"
            "  - 'score': an integer from 1 to 5\n"
            "  - 'reason': a short 1-sentence summary\n"
            "  - 'detailed_feedback': multi-sentence explanation of your scoring decision\n"
            "Do not include markdown formatting or any other text outside the JSON object."
        )
    else:
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
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.0,
        )

        judge_text = response.choices[0].message.content.strip()

        try:
            judge_text_clean = extract_code_block(judge_text)
            result = json.loads(judge_text_clean)

            d = {
                "score":        result.get("score", 0),
                "reason":       result.get("reason", "No reason provided by judge"),
                "model_output": target_response,
                "output_type":  "json",
                "judge_model":  judge_model,
                "judge_usage":  response.usage,
            }

            if include_detail:
                fb = extract_judge_feedback(result)
                d["judge_reasoning"]       = fb
                d["judge_rubric_feedback"] = fb

            return d

        except json.JSONDecodeError:
            return {
                "score":        0,
                "reason":       f"Judge failed to return valid JSON. Raw: {judge_text[:80]}",
                "model_output": target_response,
                "output_type":  "json",
                "judge_model":  judge_model,
            }

    except Exception as e:
        return {
            "score":        0,
            "reason":       f"Judge API call failed: {str(e)}",
            "model_output": target_response,
            "output_type":  "json",
            "judge_model":  judge_model,
        }
