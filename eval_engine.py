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
    return f"{entry['service']}::{entry['model']}::{entry['test_id']}"


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
# Main evaluation loop
# ---------------------------------------------------------------------------

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
):
    tracker = TokenTracker()

    # Resolve target services: CLI override → config default
    services_to_run = target_services if target_services else TARGET_SERVICES

    # 1. Load Dataset
    try:
        with open(dataset_path, "r") as f:
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
