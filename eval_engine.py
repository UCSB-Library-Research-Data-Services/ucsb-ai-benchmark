import json
import time
import logging
import os
import base64
from io import BytesIO
from PIL import Image
from config import SERVICES
from eval_config import TARGET_SERVICES, DEFAULT_JUDGE_SERVICE, DEFAULT_JUDGE_MODEL, SERVICE_BUDGETS, EXCLUDED_MODELS
from eval_scorers import (
    score_programmatic_code,
    score_programmatic_metadata,
    score_exact_match,
    score_with_judge,
    score_ocr_table,
    score_document_extraction,
)
from token_tracker import TokenTracker
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Default budget used when a service is not listed in SERVICE_BUDGETS.
_DEFAULT_BUDGET = 500_000

# Retry settings
_MAX_RETRIES = 2
_RETRY_BASE_DELAY = 5

# Categories that require image input — used to skip non-vision models gracefully.
_VISION_CATEGORIES = {"ocr", "vision_qa", "document_extraction"}

# Known model name fragments that identify vision-capable models.
# If a task requires vision and the model does NOT match any of these,
# it is skipped with a clear log message rather than failing with an API error.
_VISION_MODEL_HINTS = [
    "llava", "llava-phi", "moondream", "bakllava",   # Ollama vision
    "vision", "vl", "multimodal",                     # generic hints
    "gpt-4o", "gpt-4-vision",                         # OpenAI
    "claude-3", "claude-4",                           # Anthropic (all support vision)
    "gemini",                                          # Google (all support vision)
    "pixtral", "mistral-large",                       # Mistral vision models
    "phi-3-vision", "phi-4", "phi4",                  # Microsoft Phi vision
    "qwen-vl", "qwen2-vl", "internvl",               # Qwen/InternVL vision
    "idefics",                                         # HuggingFace vision
    "all-the-things",                                  # UCSB multi-capability model
]

# Judge-only categories: tasks whose score comes entirely from the LLM judge.
# When a model is the judge, it MUST NOT also be evaluated on these categories —
# grading your own output inflates scores and violates evaluation validity.
_JUDGE_SCORED_CATEGORIES = {"literature_synthesis", "ocr", "document_extraction", "vision_qa"}


# ---------------------------------------------------------------------------
# Judge / evaluated-model conflict detection
# ---------------------------------------------------------------------------

def _normalise_model_id(model_id: str) -> str:
    """Lowercase and strip common version suffixes for family matching."""
    return model_id.lower().strip()


def _is_judge_conflict(
    judge_service: str,
    judge_model: str,
    target_service: str,
    target_model: str,
) -> tuple[bool, str]:
    """Return (True, reason) if the evaluated model would be judging itself.

    Three conflict levels are checked, from strictest to loosest:

    1. EXACT — same service AND same model ID (clear self-grading).
    2. CROSS-SERVICE EXACT — different service but identical model name
       (e.g. judge=aicommons/claude-v4.6-sonnet, target=CIT/claude-v4.6-sonnet).
    3. FAMILY — one ID is a substring of the other
       (e.g. judge=aicommons/claude-v4.6-sonnet would match
        target=aicommons/claude-v4.5-haiku because "claude" is shared,
        so we use a stricter check: the *full versioned name* must be a
        substring, not just the family prefix).

    The family check is intentionally conservative to avoid over-blocking:
    "gemini-3.5-flash" judging "gemini-3.6-flash" are different models from
    different checkpoints — we allow that. But "claude-v4.6-sonnet" judging
    "claude-v4.6-sonnet" (exact cross-service) is blocked.
    """
    j = _normalise_model_id(judge_model)
    t = _normalise_model_id(target_model)

    # Level 1: exact same service + model
    if judge_service == target_service and j == t:
        return True, (
            f"EXACT SELF-JUDGE: judge and target are the same model "
            f"({target_service}/{target_model})."
        )

    # Level 2: exact model name match across different services
    if j == t:
        return True, (
            f"CROSS-SERVICE SELF-JUDGE: judge model '{judge_model}' "
            f"matches target '{target_service}/{target_model}' by name."
        )

    # Level 3: versioned family match — full judge name is a substring of
    # target (or vice-versa). This catches e.g. gpt-4o judging gpt-4.
    # We require the match to span at least 6 chars to avoid false positives
    # from short common tokens ("llm", "gpt", etc.).
    MIN_MATCH_LEN = 6
    if len(j) >= MIN_MATCH_LEN and (j in t or t in j):
        return True, (
            f"FAMILY SELF-JUDGE: judge '{judge_model}' and target "
            f"'{target_service}/{target_model}' share the same model family."
        )

    return False, ""


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def encode_image_to_base64(image_path: str, max_size: int = 1024) -> str:
    """Read an image, resize if too large, and return a base64-encoded JPEG string.

    Resizing to max_size prevents '413 Request Entity Too Large' errors from
    API gateways. Converts RGBA/P images to RGB before JPEG encoding.
    """
    with Image.open(image_path) as img:
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        w, h = img.size
        if w > max_size or h > max_size:
            if w > h:
                new_w, new_h = max_size, int(max_size * h / w)
            else:
                new_h, new_w = max_size, int(max_size * w / h)
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")


def _is_vision_model(model_id: str) -> bool:
    """Heuristic: return True if the model name suggests vision capability."""
    lower = model_id.lower()
    return any(hint in lower for hint in _VISION_MODEL_HINTS)


def _build_message_content(test_case: dict) -> str | list:
    """Build the message content for a chat completion call.

    For text-only tasks returns a plain string.
    For tasks with an image_path, returns the multimodal content list
    (text + base64-encoded image) following the OpenAI vision API format.
    """
    image_path = test_case.get("image_path")
    if not image_path:
        return test_case["prompt"]

    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Image not found: '{image_path}'. "
            f"Ensure the path is relative to the project root."
        )

    b64 = encode_image_to_base64(image_path)
    return [
        {"type": "text", "text": test_case["prompt"]},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
    ]


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def get_client(service_name: str) -> OpenAI:
    config = SERVICES.get(service_name)
    if not config or not config.get("key"):
        raise ValueError(f"Service '{service_name}' is missing or has no API key.")
    return OpenAI(
        base_url=config.get("url"),
        api_key=config.get("key"),
        timeout=120.0,
    )


def _call_with_retry(client: OpenAI, model_id: str, messages: list) -> object:
    """Chat completion call with exponential-backoff retry."""
    last_exc = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            return client.chat.completions.create(model=model_id, messages=messages)
        except Exception as exc:
            last_exc = exc
            if attempt < _MAX_RETRIES:
                delay = _RETRY_BASE_DELAY * (2 ** attempt)
                logging.warning(
                    f"      -> API call failed (attempt {attempt + 1}/{_MAX_RETRIES + 1}): {exc}. "
                    f"Retrying in {delay}s..."
                )
                time.sleep(delay)
    raise last_exc


# ---------------------------------------------------------------------------
# Results log helpers
# ---------------------------------------------------------------------------

def _load_existing_results(path: str) -> list:
    try:
        with open(path, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return []


def _make_run_key(entry: dict) -> str:
    """Unique key identifying a (service, model, test_id [, judge_model]) combination.

    Judge-awareness:
    - Entries WITHOUT a 'judge_model' field (programmatic scorers, or legacy entries
      pre-dating this change) use the 3-part key: service::model::test_id
    - Entries WITH a 'judge_model' field use the 4-part key:
      service::model::test_id::judge_model

    This allows the same (service, model, test_id) to be stored multiple times —
    once per judge — enabling side-by-side judge comparison without overwriting.

    Backward compatibility: all existing entries without 'judge_model' are
    addressed by their 3-part key, exactly as before.
    """
    base = f"{entry['service']}::{entry['model']}::{entry['test_id']}"
    judge = entry.get("judge_model")
    return f"{base}::{judge}" if judge else base


def _append_result(result: dict, log_path: str = "results_log.json") -> None:
    """Atomically append one result — idempotent on duplicate keys."""
    try:
        existing = _load_existing_results(log_path)
        run_key = _make_run_key(result)
        if not any(_make_run_key(r) == run_key for r in existing):
            existing.append(result)
            with open(log_path, "w") as f:
                json.dump(existing, f, indent=2)
    except Exception as e:
        logging.error(f"Failed to append result to {log_path}: {e}")


# ---------------------------------------------------------------------------
# Re-judge pipeline
# ---------------------------------------------------------------------------

# Categories whose scores come exclusively from the LLM judge.
# Only these entries can be re-judged (programmatic scores are deterministic
# and do not need a second opinion).
_RE_JUDGEABLE_CATEGORIES = {
    "literature_synthesis",
    "ocr",
    "document_extraction",
    "vision_qa",
}


def re_judge_from_log(
    judge_service: str,
    judge_model: str,
    source_judge: str | None = None,
    categories: list[str] | None = None,
    services: list[str] | None = None,
    include_judge_detail: bool = False,
    allow_self_judge: bool = False,
    log_path: str = "results_log.json",
):
    """Re-score existing log entries with a different judge, without re-calling targets.

    Reads `model_output` from already-completed log entries and re-sends each
    one to the new judge. Stores results under a judge-aware key so they coexist
    alongside the original scores — the original entries are never modified.

    Args:
        judge_service:      Service hosting the new judge (e.g. "aicommons").
        judge_model:        New judge model ID (e.g. "claude-v4.6-sonnet").
        source_judge:       If set, only re-judge entries previously scored by
                            this judge (e.g. "gemini-3.5-flash"). None = all.
        categories:         Restrict to these categories. None = all re-judgeable.
        services:           Restrict to entries from these target services. None = all.
        include_judge_detail: Request multi-sentence reasoning from judge.
        allow_self_judge:   If False (default), skip entries where the new judge
                            matches the evaluated model (conflict guard).
        log_path:           Path to results_log.json.
    """
    tracker = TokenTracker()

    # Initialise new judge client
    logging.info(f"Re-judge run — new judge: {judge_service} / {judge_model}")
    try:
        judge_client = get_client(judge_service)
    except Exception as e:
        logging.error(f"Failed to initialise judge: {e}")
        return

    # Load existing results
    existing = _load_existing_results(log_path)
    if not existing:
        logging.warning(f"No existing results found in '{log_path}'.")
        return

    # Build set of already-completed re-judge keys to support resume
    completed_keys = {_make_run_key(r) for r in existing}

    # Filter to candidates
    target_cats = set(categories) if categories else _RE_JUDGEABLE_CATEGORIES
    candidates = [
        r for r in existing
        if r["category"] in target_cats
        and r.get("model_output")               # must have stored output
        and r.get("score") is not None          # must have a real score
        and not r.get("judge_conflict")         # skip already-conflicted entries
        and (source_judge is None or r.get("judge_model") == source_judge)
        and (services is None or r["service"] in services)
    ]

    logging.info(
        f"Found {len(candidates)} candidate entries to re-judge "
        f"(source_judge={source_judge!r}, categories={sorted(target_cats)}, "
        f"services={services})"
    )

    if not candidates:
        logging.warning("Nothing to re-judge. Check --source-judge / --categories / --services filters.")
        return

    new_count = 0
    skipped_conflict = 0
    skipped_existing = 0

    for orig in candidates:
        svc_name   = orig["service"]
        model_id   = orig["model"]
        test_id    = orig["test_id"]
        category   = orig["category"]

        # Build the judge-aware key for the NEW judge
        candidate_entry_key_fields = {
            "service": svc_name, "model": model_id,
            "test_id": test_id,  "judge_model": judge_model,
        }
        new_key = _make_run_key(candidate_entry_key_fields)

        if new_key in completed_keys:
            logging.info(f"    Skipping {svc_name}/{model_id}/{test_id} — already re-judged by {judge_model}.")
            skipped_existing += 1
            continue

        # Judge-conflict guard
        if not allow_self_judge and category in _JUDGE_SCORED_CATEGORIES:
            conflict, reason = _is_judge_conflict(judge_service, judge_model, svc_name, model_id)
            if conflict:
                logging.warning(
                    f"    Skipping {svc_name}/{model_id}/{test_id} — "
                    f"judge conflict: {reason}"
                )
                skipped_conflict += 1
                continue

        logging.info(f"  Re-judging {svc_name}/{model_id} — {test_id} ({category})")

        # Retrieve what the original evaluation task looked like
        # We need prompt + rubric from the dataset, so we try to reconstruct them
        # from the original entry; if not present fall back to generic rubric.
        original_prompt = orig.get("prompt", "")   # not stored yet — see note below
        original_rubric = orig.get("judge_rubric", "")

        # If prompt/rubric aren't stored in the log (current schema doesn't store them),
        # load them from the dataset file that was originally used.
        if not original_prompt or not original_rubric:
            original_prompt, original_rubric = _lookup_prompt_rubric(test_id)

        model_output = orig["model_output"]

        judge_budget = SERVICE_BUDGETS.get(judge_service, _DEFAULT_BUDGET)
        if not tracker.check_budget(judge_service, judge_budget):
            logging.warning(f"Judge budget exceeded for {judge_service}. Stopping re-judge run.")
            break

        try:
            if category == "document_extraction":
                score_result = score_document_extraction(
                    response_text=model_output,
                    judge_client=judge_client,
                    judge_model=judge_model,
                    prompt=original_prompt,
                    rubric=original_rubric,
                    tracker=tracker,
                    include_detail=include_judge_detail,
                )
            else:
                score_result = score_with_judge(
                    target_response=model_output,
                    prompt=original_prompt,
                    rubric=original_rubric,
                    judge_client=judge_client,
                    judge_model=judge_model,
                    tracker=tracker,
                    include_detail=include_judge_detail,
                )

            if "judge_usage" in score_result:
                tracker.add_usage(
                    judge_service, judge_model,
                    score_result.pop("judge_usage"), autosave=False
                )

            # Build new log entry — carries all original perf metrics,
            # plus the new judge's score and reasoning
            new_entry = {
                # Identity
                "service":           svc_name,
                "model":             model_id,
                "test_id":           test_id,
                "category":          category,
                # Original performance metrics (unchanged)
                "latency_sec":       orig.get("latency_sec", 0),
                "tps":               orig.get("tps", 0),
                "prompt_tokens":     orig.get("prompt_tokens", 0),
                "completion_tokens": orig.get("completion_tokens", 0),
                # New judge's assessment
                "score":             score_result["score"],
                "reason":            score_result["reason"],
                "judge_model":       judge_model,
                # Provenance: link back to the original entry
                "re_judged_from":    orig.get("judge_model", "unknown"),
                # Original model output (preserved for further re-judging)
                "model_output":      model_output,
                "output_type":       orig.get("output_type", ""),
            }
            # Optional detail fields
            for k in ("judge_reasoning", "judge_rubric_feedback",
                      "ocr_accuracy", "fields_correct", "fields_total"):
                if k in score_result:
                    new_entry[k] = score_result[k]

            _append_result(new_entry, log_path)
            completed_keys.add(new_key)
            new_count += 1

            logging.info(
                f"    [{orig.get('judge_model','?')} → {judge_model}] "
                f"score: {orig['score']} → {score_result['score']} | "
                f"{orig['reason'][:40]}... | {score_result['reason'][:40]}..."
            )

        except Exception as e:
            logging.error(f"    Error re-judging {svc_name}/{model_id}/{test_id}: {e}")

    # Flush token usage
    tracker.save()

    logging.info(f"\n{'='*50}")
    logging.info(
        f"Re-judge complete — {new_count} new entries, "
        f"{skipped_existing} already done, "
        f"{skipped_conflict} conflict-skipped."
    )
    logging.info(f"{'='*50}")


def _lookup_prompt_rubric(test_id: str) -> tuple[str, str]:
    """Look up the original prompt and judge_rubric for a test_id.

    Searches all known dataset files in order of specificity.
    Returns ("", "") if not found — the judge will still work but
    context will be minimal.
    """
    dataset_files = [
        "evaluations/sci_code_problems_dev.jsonl",
        "evaluations/eval_dataset_multimodal.jsonl",
        "evaluations/eval_dataset_v2.jsonl",
        "evaluations/eval_dataset.jsonl",
        "eval_dataset_multimodal.json",
        "eval_dataset_v2.json",
        "eval_dataset.json",
    ]
    for path in dataset_files:
        try:
            with open(path) as f:
                if path.endswith(".jsonl"):
                    for line in f:
                        if not line.strip(): continue
                        task = json.loads(line)
                        if task.get("problem_id") == test_id:
                            rubric = "Score 1-5 for accuracy and completeness."
                            if task.get("sub_steps") and len(task["sub_steps"]) > 0:
                                rubric = task["sub_steps"][0].get("judge_rubric", rubric)
                            return (task.get("problem_description_main", ""), rubric)
                else:
                    for task in json.load(f):
                        if task["id"] == test_id:
                            return (
                                task.get("prompt", ""),
                                task.get("judge_rubric", "Score 1-5 for accuracy and completeness."),
                            )
        except (FileNotFoundError, json.JSONDecodeError):
            continue
    logging.warning(f"Could not find prompt/rubric for test_id '{test_id}' in any dataset file.")
    return "", "Score the model output 1-5 based on accuracy and completeness."

def run_evaluation(
    dataset_path: str = "eval_dataset.json",
    target_services: list[str] | None = None,
    test_run: bool = False,
    judge_service: str = DEFAULT_JUDGE_SERVICE,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    force_all_models: bool = False,
    include_model_output: bool = False,
    include_judge_detail: bool = False,
    resume: bool = False,
    allow_self_judge: bool = False,
):
    tracker = TokenTracker()

    # Resolve target services: CLI override → config default
    services_to_run = target_services if target_services else TARGET_SERVICES

    # 1. Load Dataset
    try:
        with open(dataset_path, "r") as f:
            if dataset_path.endswith(".jsonl"):
                raw_dataset = [json.loads(line) for line in f if line.strip()]
                dataset = []
                for item in raw_dataset:
                    flat_item = {
                        "id": item.get("problem_id", ""),
                        "category": item.get("problem_name", ""),
                        "prompt": item.get("problem_description_main", "")
                    }
                    if item.get("sub_steps") and len(item["sub_steps"]) > 0:
                        sub = item["sub_steps"][0]
                        for k, v in sub.items():
                            if k not in ["step_number", "step_description_prompt", "step_background"]:
                                flat_item[k] = v
                    dataset.append(flat_item)
            else:
                dataset = json.load(f)
        logging.info(f"Loaded {len(dataset)} task(s) from '{dataset_path}'")
    except Exception as e:
        logging.error(f"Failed to load dataset '{dataset_path}': {e}")
        return

    if test_run:
        dataset = dataset[:1]
        logging.info("TEST RUN MODE: Only running first task.")

    # Detect whether any task in this dataset requires vision
    dataset_needs_vision = any(
        t.get("category") in _VISION_CATEGORIES or t.get("image_path")
        for t in dataset
    )

    # 2. Initialize Judge Client
    logging.info(f"Initializing Judge: {judge_service} / {judge_model}")
    try:
        judge_client = get_client(judge_service)
    except Exception as e:
        logging.error(f"Failed to initialize judge: {e}")
        return

    # 3. Load existing results / resume
    results_log_path = "results_log.json"
    completed_keys: set[str] = set()

    if resume:
        logging.info("RESUME MODE: skipping already-completed triplets.")
        existing = _load_existing_results(results_log_path)
        completed_keys = {_make_run_key(r) for r in existing}
        logging.info(f"  Found {len(completed_keys)} completed triplet(s).")
    else:
        if os.path.exists(results_log_path) and not test_run:
            logging.info(
                "Fresh run: appending to existing results_log.json "
                "(use --resume to skip already-completed entries)"
            )

    new_results_count = 0

    # 4. Iterate Target Services
    for svc_name in services_to_run:
        logging.info(f"\n--- Evaluating Service: {svc_name} ---")

        service_budget = SERVICE_BUDGETS.get(svc_name, _DEFAULT_BUDGET)
        if not tracker.check_budget(svc_name, service_budget):
            logging.warning(f"Skipping {svc_name} — budget limit reached.")
            continue

        try:
            target_client = get_client(svc_name)
            active_models = [m.id for m in target_client.models.list().data]
        except Exception as e:
            logging.error(f"Failed to connect to {svc_name}: {e}")
            continue

        # 5. Iterate Models
        evaluated_count = 0
        for model_id in active_models:
            if not force_all_models:
                if model_id in EXCLUDED_MODELS:
                    logging.info(f"  Skipping {model_id} (exclusion list)")
                    continue
                if any(x in model_id.lower() for x in ["embedding", "reranker", "cloud"]):
                    logging.info(f"  Skipping {model_id} (exclusion pattern)")
                    continue

            # Graceful skip: if the dataset has vision tasks and this model
            # doesn't appear to support images, skip the whole model.
            if dataset_needs_vision and not _is_vision_model(model_id):
                # Only skip the model entirely if ALL tasks in the dataset require vision.
                # If the dataset is mixed (text + vision), we skip only the vision tasks below.
                all_vision = all(
                    t.get("category") in _VISION_CATEGORIES or t.get("image_path")
                    for t in dataset
                )
                if all_vision:
                    logging.info(
                        f"  Skipping {model_id} — vision-only dataset and "
                        f"model does not appear vision-capable."
                    )
                    continue

            logging.info(f"  Testing Model: {model_id}")
            evaluated_count += 1

            for test_case in dataset:
                run_key = f"{svc_name}::{model_id}::{test_case['id']}"

                if run_key in completed_keys:
                    logging.info(f"    Skipping {test_case['id']} — already recorded.")
                    continue

                # Per-task vision skip: if this specific task needs vision
                # and the model doesn't support it, record a clean skip entry.
                task_needs_vision = (
                    test_case.get("category") in _VISION_CATEGORIES
                    or bool(test_case.get("image_path"))
                )
                if task_needs_vision and not _is_vision_model(model_id):
                    logging.info(
                        f"    Skipping {test_case['id']} for {model_id} "
                        f"— vision task, model not vision-capable."
                    )
                    skip_entry = {
                        "service": svc_name, "model": model_id,
                        "test_id": test_case["id"], "category": test_case["category"],
                        "latency_sec": 0, "tps": 0,
                        "prompt_tokens": 0, "completion_tokens": 0,
                        "score": 0,
                        "reason": "Skipped: task requires vision but model does not support image input.",
                    }
                    _append_result(skip_entry)
                    completed_keys.add(run_key)
                    continue

                # Judge-conflict skip: a model must never grade its own output.
                # Only applies to judge-scored categories (subjective tasks).
                # Programmatic tasks (code, metadata, exact-match) are unaffected.
                if (not allow_self_judge
                        and test_case.get("category") in _JUDGE_SCORED_CATEGORIES):
                    conflict, reason = _is_judge_conflict(
                        judge_service, judge_model, svc_name, model_id
                    )
                    if conflict:
                        logging.warning(
                            f"    Skipping {test_case['id']} for {model_id} "
                            f"— judge conflict: {reason}"
                        )
                        conflict_entry = {
                            "service": svc_name, "model": model_id,
                            "test_id": test_case["id"], "category": test_case["category"],
                            "latency_sec": 0, "tps": 0,
                            "prompt_tokens": 0, "completion_tokens": 0,
                            "score": None,
                            "reason": f"Skipped: judge conflict — {reason}",
                            "judge_conflict": True,
                        }
                        _append_result(conflict_entry)
                        completed_keys.add(run_key)
                        continue

                logging.info(f"    Task: {test_case['category']} ({test_case['id']})")

                start_time = time.time()
                try:
                    message_content = _build_message_content(test_case)
                    response = _call_with_retry(
                        target_client, model_id,
                        [{"role": "user", "content": message_content}],
                    )
                    latency = time.time() - start_time

                    target_text = response.choices[0].message.content
                    tracker.add_usage(svc_name, model_id, response.usage, autosave=False)

                    prompt_tokens     = getattr(response.usage, "prompt_tokens", 0)
                    completion_tokens = getattr(response.usage, "completion_tokens", 0)
                    tps = completion_tokens / latency if latency > 0 else 0

                    # 6. Score the Response
                    score_result = {"score": 0, "reason": "Scorer not matched"}
                    category = test_case["category"]

                    if category == "code_generation":
                        score_result = score_programmatic_code(
                            target_text,
                            dummy_csv=test_case.get(
                                "dummy_csv",
                                "temperature,pressure\n15,1000\n25,1010\n30,1020\n",
                            ),
                            expected_output=test_case.get("expected_output"),
                            include_output=include_model_output,
                        )

                    elif category == "metadata_generation":
                        score_result = score_programmatic_metadata(
                            target_text,
                            test_case["expected_schema_keys"],
                            expected_values=test_case.get("expected_values"),
                            include_output=include_model_output,
                        )

                    elif category == "data_interpretation":
                        score_result = score_exact_match(
                            target_text,
                            test_case["expected_exact_match"],
                            include_output=include_model_output,
                        )

                    elif category == "vision_qa":
                        # Some vision_qa tasks have a deterministic answer;
                        # others use expected_exact_match == "judge" to fall through.
                        expected = test_case.get("expected_exact_match", "")
                        if expected and expected.lower() != "judge":
                            score_result = score_exact_match(
                                target_text, expected,
                                include_output=include_model_output,
                            )
                        else:
                            # Non-deterministic vision QA → judge
                            judge_budget = SERVICE_BUDGETS.get(judge_service, _DEFAULT_BUDGET)
                            if tracker.check_budget(judge_service, judge_budget):
                                score_result = score_with_judge(
                                    target_response=target_text,
                                    prompt=test_case["prompt"],
                                    rubric=test_case.get("judge_rubric", "Score 1-5 for accuracy."),
                                    judge_client=judge_client,
                                    judge_model=judge_model,
                                    tracker=tracker,
                                    include_detail=include_judge_detail,
                                )
                                if "judge_usage" in score_result:
                                    tracker.add_usage(judge_service, judge_model, score_result.pop("judge_usage"), autosave=False)
                            else:
                                score_result = {"score": 0, "reason": "Judge budget exceeded."}

                    elif category == "ocr":
                        ground_truth = test_case.get("ground_truth_rows", [])
                        key_fields   = test_case.get("key_fields", [])
                        if ground_truth and key_fields:
                            # Deterministic OCR table — use programmatic scorer
                            score_result = score_ocr_table(
                                target_text,
                                ground_truth_rows=ground_truth,
                                key_fields=key_fields,
                                include_output=include_model_output,
                            )
                            # If programmatic score is low, supplement with a judge check
                            if score_result["score"] < 4:
                                judge_budget = SERVICE_BUDGETS.get(judge_service, _DEFAULT_BUDGET)
                                if tracker.check_budget(judge_service, judge_budget):
                                    judge_result = score_with_judge(
                                        target_response=target_text,
                                        prompt=test_case["prompt"],
                                        rubric=test_case.get("judge_rubric", "Score 1-5 for OCR accuracy."),
                                        judge_client=judge_client,
                                        judge_model=judge_model,
                                        tracker=tracker,
                                        include_detail=include_judge_detail,
                                    )
                                    if "judge_usage" in judge_result:
                                        tracker.add_usage(judge_service, judge_model, judge_result.pop("judge_usage"), autosave=False)
                                    # Take the higher of the two scores (benefit of the doubt)
                                    if judge_result["score"] > score_result["score"]:
                                        score_result["score"]  = judge_result["score"]
                                        score_result["reason"] = (
                                            f"[Programmatic: {score_result['reason'][:60]}] "
                                            f"[Judge: {judge_result['reason']}]"
                                        )
                                        for k in ("judge_reasoning", "judge_rubric_feedback", "judge_model"):
                                            if k in judge_result:
                                                score_result[k] = judge_result[k]
                        else:
                            # No ground truth — pure judge evaluation
                            judge_budget = SERVICE_BUDGETS.get(judge_service, _DEFAULT_BUDGET)
                            if tracker.check_budget(judge_service, judge_budget):
                                score_result = score_with_judge(
                                    target_response=target_text,
                                    prompt=test_case["prompt"],
                                    rubric=test_case.get("judge_rubric", "Score 1-5 for OCR accuracy."),
                                    judge_client=judge_client,
                                    judge_model=judge_model,
                                    tracker=tracker,
                                    include_detail=include_judge_detail,
                                )
                                if "judge_usage" in score_result:
                                    tracker.add_usage(judge_service, judge_model, score_result.pop("judge_usage"), autosave=False)
                            else:
                                score_result = {"score": 0, "reason": "Judge budget exceeded."}

                    elif category == "literature_synthesis":
                        judge_budget = SERVICE_BUDGETS.get(judge_service, _DEFAULT_BUDGET)
                        if tracker.check_budget(judge_service, judge_budget):
                            score_result = score_with_judge(
                                target_response=target_text,
                                prompt=test_case["prompt"],
                                rubric=test_case["judge_rubric"],
                                judge_client=judge_client,
                                judge_model=judge_model,
                                tracker=tracker,
                                include_detail=include_judge_detail,
                            )
                            if "judge_usage" in score_result:
                                tracker.add_usage(judge_service, judge_model, score_result.pop("judge_usage"), autosave=False)
                        else:
                            score_result = {"score": 0, "reason": "Judge budget exceeded."}

                    elif category == "document_extraction":
                        judge_budget = SERVICE_BUDGETS.get(judge_service, _DEFAULT_BUDGET)
                        if tracker.check_budget(judge_service, judge_budget):
                            score_result = score_document_extraction(
                                response_text=target_text,
                                judge_client=judge_client,
                                judge_model=judge_model,
                                prompt=test_case["prompt"],
                                rubric=test_case.get("judge_rubric", "Score 1-5 for extraction completeness and accuracy."),
                                tracker=tracker,
                                include_detail=include_judge_detail,
                            )
                            if "judge_usage" in score_result:
                                tracker.add_usage(judge_service, judge_model, score_result.pop("judge_usage"), autosave=False)
                        else:
                            score_result = {"score": 0, "reason": "Judge budget exceeded."}

                    # 7. Build log entry
                    log_entry = {
                        "service":           svc_name,
                        "model":             model_id,
                        "test_id":           test_case["id"],
                        "category":          category,
                        "latency_sec":       round(latency, 2),
                        "tps":               round(tps, 2),
                        "prompt_tokens":     prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "score":             score_result["score"],
                        "reason":            score_result["reason"],
                    }
                    # Optional transparency fields
                    for key in ("model_output", "output_type", "judge_reasoning",
                                "judge_rubric_feedback", "judge_model",
                                "ocr_accuracy", "fields_correct", "fields_total"):
                        if key in score_result:
                            log_entry[key] = score_result[key]
                    # Code execution stdout
                    if "output" in score_result:
                        log_entry["code_output"] = score_result["output"]

                    _append_result(log_entry)
                    completed_keys.add(run_key)
                    new_results_count += 1

                    logging.info(
                        f"      -> Score: {score_result['score']}/5 | TPS: {round(tps, 2)}"
                    )

                except Exception as e:
                    logging.error(
                        f"      -> Error on {model_id} / {test_case['id']}: {e}"
                    )

            # Flush token tracker once per model
            tracker.save()

            if test_run and evaluated_count >= 2:
                logging.info(f"  Reached test-run limit for {svc_name}.")
                break

    # 8. Final summary
    logging.info(f"\n{'='*50}")
    logging.info(f"Evaluation complete — {new_results_count} new result(s) recorded.")
    logging.info(f"Results saved to '{results_log_path}'")
    logging.info(f"{'='*50}")
