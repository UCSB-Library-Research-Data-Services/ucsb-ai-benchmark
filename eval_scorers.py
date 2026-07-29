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
    """Checks that the response matches the expected value.

    Used for both multiple-choice letters and short numeric/text OCR answers.

    For multiple-choice letters (single A–D character):
      Accepts "C", "C)", "The answer is C", "C. Temp vs Humidity"
      Rejects letters appearing mid-sentence ("C is wrong…").

    For short OCR/numeric answers (e.g. "59.0", "SE"):
      Uses _field_match for loose comparison:
        - Case-insensitive, strip punctuation
        - Comma vs period decimal ("59,0" ≈ "59.0")
        - Space-separated tokens ("59 0" ≈ "59.0" — common OCR artefact
          in historical documents where periods are read as spaces)
    """
    text   = response_text.strip()
    is_mcq = len(expected_match) == 1 and expected_match.upper() in "ABCDEFGHIJ"

    if is_mcq:
        # Multiple-choice: anchor-based pattern match
        letter = re.escape(expected_match.upper())
        pattern = re.compile(
            rf"(?:^{letter}(?:[).:\n]|$)|(?:answer(?:\s+is)?|the\s+answer\s+is)\s*:?\s*{letter}\b)",
            re.IGNORECASE,
        )
        if pattern.search(text):
            d = {"score": 5, "reason": f"Matched expected answer: {expected_match}"}
        else:
            d = {"score": 1, "reason": f"Expected '{expected_match}', got: '{text[:40]}'"}
    else:
        # Short OCR / numeric answer: use loose field matching
        # Strip the model response down to its first meaningful token
        first_token = text.split()[0].rstrip(".,;") if text.split() else text
        if _field_match(first_token, expected_match) or _field_match(text, expected_match):
            d = {"score": 5, "reason": f"Matched expected value: {expected_match}"}
        else:
            d = {"score": 1, "reason": f"Expected '{expected_match}', got: '{text[:40]}'"}

    if include_output:
        d["model_output"] = text
        d["output_type"]  = "multiple_choice"
    return d


# ---------------------------------------------------------------------------
# Judge helpers
# ---------------------------------------------------------------------------

def _normalise_judge_response(raw: dict) -> dict:
    """Flatten and normalise a judge JSON response into a canonical form.

    Handles two known structural variations:
    1. Flat (expected):
         {"score": 4, "reason": "...", "detailed_feedback": "..."}
    2. Nested envelope (Claude sometimes wraps output):
         {"evaluation": {"score": 4, "reasoning": "..."}}

    Also normalises key aliases:
    - "reasoning" → "reason"
    - "feedback"  → "detailed_feedback"
    - "rationale" → "reason"
    """
    # Unwrap single-key envelope (e.g. {"evaluation": {...}})
    if len(raw) == 1:
        inner = next(iter(raw.values()))
        if isinstance(inner, dict) and "score" in inner:
            raw = inner

    # Normalise key aliases
    alias_map = {
        "reasoning":        "reason",
        "rationale":        "reason",
        "explanation":      "reason",
        "feedback":         "detailed_feedback",
        "detailed_reason":  "detailed_feedback",
    }
    normalised = {}
    for k, v in raw.items():
        normalised[alias_map.get(k, k)] = v
    return normalised


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
            raw = json.loads(judge_text_clean)

            # Normalise the judge response — different models wrap their output
            # differently.  Claude sometimes returns:
            #   {"evaluation": {"score": 4, "reasoning": "..."}}
            # instead of the flat {"score": 4, "reason": "..."} we asked for.
            # We flatten one level of nesting and normalise key aliases.
            result = _normalise_judge_response(raw)

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


# ---------------------------------------------------------------------------
# Multimodal scorers
# ---------------------------------------------------------------------------

def _normalize_field(v: str) -> str:
    """Lowercase, strip whitespace, collapse multiple spaces, remove punctuation
    that varies between OCR readings (commas in numbers, trailing periods)."""
    v = str(v).lower().strip()
    v = re.sub(r"\s+", " ", v)
    return v


def _field_match(extracted: str, expected: str, allow_partial: bool = True) -> bool:
    """Loose field-level match for OCR comparison.

    Handles:
    - Case and whitespace differences
    - Trailing/leading punctuation (periods, commas)
    - Partial containment for long values (allow_partial=True)
    - Numeric equivalence: "57.5" ≈ "57,5" (comma vs period decimal)
    """
    e = _normalize_field(extracted).strip(".,;")
    x = _normalize_field(expected).strip(".,;")
    # direct match
    if e == x:
        return True
    # numeric: normalise decimal separators
    e_num = e.replace(",", ".")
    x_num = x.replace(",", ".")
    if e_num == x_num:
        return True
    # partial containment (useful for long cell values)
    if allow_partial and (x in e or e in x):
        return True
    return False


def score_ocr_table(
    response_text: str,
    ground_truth_rows: list[dict],
    key_fields: list[str],
    include_output: bool = False,
) -> dict:
    """Score an OCR table extraction against a ground-truth row list.

    This scorer is designed for structured table extraction from images
    (e.g. a 19th-century meteorological table). It does NOT re-execute code —
    it parses the model's text output and compares against known ground truth.

    The model is expected to output either:
    - A JSON array of row objects, OR
    - A markdown table (which we parse into rows)

    Scoring rubric:
        - Computes per-field accuracy across all rows × key_fields
        - Maps accuracy to 1–5 scale:
            5/5  ≥ 90% fields correct
            4/5  ≥ 70%
            3/5  ≥ 50%
            2/5  ≥ 30%
            1/5  < 30% (or unparseable output)

    Args:
        response_text:     The model's raw response string.
        ground_truth_rows: List of dicts, one per expected table row.
                           Keys must include all entries in key_fields.
        key_fields:        Which dict keys to check (subset of ground_truth_rows keys).
        include_output:    Whether to include model_output in the result.
    """
    text = response_text.strip()
    n_rows = len(ground_truth_rows)
    n_fields = len(key_fields)
    total_checks = n_rows * n_fields

    # ── Step 1: Try to parse the model response into rows ─────────────────
    extracted_rows: list[dict] = []

    # Attempt 1: JSON array
    try:
        cleaned = extract_code_block(text)
        parsed = json.loads(cleaned)
        if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], dict):
            extracted_rows = parsed
    except (json.JSONDecodeError, TypeError):
        pass

    # Attempt 2: Markdown table  (| col1 | col2 | ... |)
    if not extracted_rows:
        md_rows = []
        headers: list[str] = []
        for line in text.splitlines():
            line = line.strip()
            if not line.startswith("|"):
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            if all(re.match(r"^[-:]+$", c) for c in cells if c):
                continue  # separator row
            if not headers:
                headers = [c.lower().replace(" ", "_") for c in cells]
            else:
                if len(cells) == len(headers):
                    md_rows.append(dict(zip(headers, cells)))
        extracted_rows = md_rows

    # If still nothing parseable, return score 1
    if not extracted_rows:
        d = {
            "score":  1,
            "reason": "Could not parse any rows from the model output (not JSON array or markdown table).",
            "ocr_accuracy": 0.0,
            "fields_correct": 0,
            "fields_total": total_checks,
        }
        if include_output:
            d["model_output"] = text
            d["output_type"]  = "ocr_table"
        return d

    # ── Step 2: Align extracted rows to ground-truth rows ─────────────────
    # Use the first key_field as an anchor for row matching (e.g. "epoca" / time-of-day).
    # For each ground-truth row, find the best-matching extracted row.
    anchor = key_fields[0]
    fields_correct = 0

    for gt_row in ground_truth_rows:
        gt_anchor_val = gt_row.get(anchor, "")
        # Find best matching extracted row by anchor field
        best_row = None
        for ex_row in extracted_rows:
            # Try all key variations of the anchor
            for k, v in ex_row.items():
                if _field_match(str(v), gt_anchor_val):
                    best_row = ex_row
                    break
            if best_row:
                break

        if best_row is None:
            # No row matched by anchor: check all key_fields against all extracted values
            # (model may have used different key names)
            all_extracted_vals = " | ".join(str(v) for v in best_row.values()) if best_row else ""
        else:
            all_extracted_vals = " | ".join(str(v) for v in best_row.values())

        for field in key_fields:
            expected_val = str(gt_row.get(field, ""))
            if best_row is None:
                # Row not found — check if the value appears anywhere in the full response
                if _field_match(text, expected_val, allow_partial=True):
                    fields_correct += 1
                # else 0
            else:
                # Check the matched row's value for this field
                extracted_val = ""
                for k, v in best_row.items():
                    if field in k.lower() or k.lower() in field:
                        extracted_val = str(v)
                        break
                if not extracted_val:
                    # Key name didn't match — try positional (nth field in row)
                    row_vals = list(best_row.values())
                    gt_vals  = list(gt_row.values())
                    if field in gt_row:
                        field_idx = list(gt_row.keys()).index(field)
                        if field_idx < len(row_vals):
                            extracted_val = str(row_vals[field_idx])

                if _field_match(extracted_val, expected_val):
                    fields_correct += 1

    accuracy = fields_correct / total_checks if total_checks > 0 else 0.0

    # ── Step 3: Map accuracy to score ────────────────────────────────────
    if accuracy >= 0.90:
        score, grade = 5, "excellent"
    elif accuracy >= 0.70:
        score, grade = 4, "good"
    elif accuracy >= 0.50:
        score, grade = 3, "partial"
    elif accuracy >= 0.30:
        score, grade = 2, "poor"
    else:
        score, grade = 1, "very poor"

    d = {
        "score":          score,
        "reason":         (
            f"OCR table accuracy: {accuracy:.0%} ({fields_correct}/{total_checks} fields correct — {grade}). "
            f"Extracted {len(extracted_rows)} row(s), expected {n_rows}."
        ),
        "ocr_accuracy":   round(accuracy, 4),
        "fields_correct": fields_correct,
        "fields_total":   total_checks,
    }
    if include_output:
        d["model_output"] = text
        d["output_type"]  = "ocr_table"
    return d


def score_document_extraction(
    response_text: str,
    judge_client: OpenAI,
    judge_model: str,
    prompt: str,
    rubric: str,
    tracker,
    include_detail: bool = False,
) -> dict:
    """Score non-deterministic structured extraction from a document image.

    Used for tasks where the content is not known in advance (e.g. the back of
    a historical catalogue card) and cannot be verified programmatically.
    Delegates to the LLM judge with a rubric focused on:
      - Presence of expected structural elements (fields, sections)
      - Internal consistency of extracted values
      - Absence of hallucinated content

    This is a thin wrapper around score_with_judge that sets the output_type
    to 'document_extraction' for dashboard filtering.
    """
    result = score_with_judge(
        target_response=response_text,
        prompt=prompt,
        rubric=rubric,
        judge_client=judge_client,
        judge_model=judge_model,
        tracker=tracker,
        include_detail=include_detail,
    )
    # Override output_type for dashboard filtering
    result["output_type"] = "document_extraction"
    return result
