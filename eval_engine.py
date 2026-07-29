import json
import time
import logging
from config import SERVICES
from eval_config import TARGET_SERVICES, DEFAULT_JUDGE_SERVICE, DEFAULT_JUDGE_MODEL, SERVICE_BUDGETS, EXCLUDED_MODELS
from eval_scorers import score_programmatic_code, score_programmatic_metadata, score_exact_match, score_with_judge
from token_tracker import TokenTracker
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Default budget used when a service is not listed in SERVICE_BUDGETS.
# Set conservatively so an unconfigured service won't silently consume tokens.
_DEFAULT_BUDGET = 500_000

# How many times to retry a failed API call before recording it as an error.
_MAX_RETRIES = 2
# Base delay in seconds between retries (doubles each attempt).
_RETRY_BASE_DELAY = 5


def get_client(service_name: str) -> OpenAI:
    config = SERVICES.get(service_name)
    if not config or not config.get("key"):
        raise ValueError(f"Service {service_name} is missing or has no API key.")
    return OpenAI(
        base_url=config.get("url"),
        api_key=config.get("key"),
        timeout=120.0
    )


def _call_with_retry(client: OpenAI, model_id: str, messages: list) -> object:
    """Calls the chat completions API with simple exponential-backoff retry.

    Returns the response object on success, or raises the last exception after
    all retries are exhausted.
    """
    last_exc = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            return client.chat.completions.create(
                model=model_id,
                messages=messages
            )
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


def _load_existing_results(path: str) -> list:
    """Loads an existing results log so new runs can be appended to it.

    Returns an empty list if the file doesn't exist or is not valid JSON.
    """
    try:
        with open(path, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return []


def _make_run_key(entry: dict) -> str:
    """Unique key that identifies a (service, model, test_id) combination."""
    return f"{entry['service']}::{entry['model']}::{entry['test_id']}"


def run_evaluation(
    test_run: bool = False,
    judge_service: str = DEFAULT_JUDGE_SERVICE,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    force_all_models: bool = False
):
    tracker = TokenTracker()

    # 1. Load Dataset
    try:
        with open("eval_dataset.json", "r") as f:
            dataset = json.load(f)
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        return

    if test_run:
        dataset = dataset[:1]  # Just run the first test
        logging.info("TEST RUN MODE: Only running 1 test case.")

    # 2. Initialize Judge Client
    logging.info(f"Initializing Judge Client: {judge_service} / {judge_model}")
    try:
        judge_client = get_client(judge_service)
    except Exception as e:
        logging.error(f"Failed to initialize judge: {e}")
        return

    # 3. Load existing results and build a set of already-completed run keys so we
    #    never overwrite a previously recorded result (append / merge behaviour).
    results_log_path = "results_log.json"
    results_log = _load_existing_results(results_log_path)
    completed_keys = {_make_run_key(r) for r in results_log}
    new_results_count = 0

    # 4. Iterate Target Services
    for target_service_name in TARGET_SERVICES:
        logging.info(f"\n--- Evaluating Service: {target_service_name} ---")

        # Guard against services missing from SERVICE_BUDGETS (1.2 fix)
        service_budget = SERVICE_BUDGETS.get(target_service_name, _DEFAULT_BUDGET)

        if not tracker.check_budget(target_service_name, service_budget):
            logging.warning(f"Skipping {target_service_name} due to budget limits.")
            continue

        try:
            target_client = get_client(target_service_name)
            models_page = target_client.models.list()
            active_models = [m.id for m in models_page.data]
        except Exception as e:
            logging.error(f"Failed to connect to {target_service_name}: {e}")
            continue

        # 5. Iterate Models & Prompts
        evaluated_count = 0
        for model_id in active_models:
            if not force_all_models:
                if model_id in EXCLUDED_MODELS:
                    logging.info(f"  Skipping Model: {model_id} (in exclusion list)")
                    continue
                if any(x in model_id.lower() for x in ["embedding", "reranker", "cloud"]):
                    logging.info(f"  Skipping Model: {model_id} (matches exclusion pattern)")
                    continue

            logging.info(f"  Testing Model: {model_id}")
            evaluated_count += 1

            for test_case in dataset:
                run_key = f"{target_service_name}::{model_id}::{test_case['id']}"

                # Skip if this exact (service, model, test) was already recorded
                if run_key in completed_keys:
                    logging.info(f"    Skipping {test_case['id']} — already in results log.")
                    continue

                logging.info(f"    Task: {test_case['category']} ({test_case['id']})")

                start_time = time.time()
                try:
                    response = _call_with_retry(
                        target_client,
                        model_id,
                        [{"role": "user", "content": test_case['prompt']}]
                    )
                    latency = time.time() - start_time

                    target_text = response.choices[0].message.content

                    # Track target token usage (flushed to disk after full model loop)
                    tracker.add_usage(target_service_name, model_id, response.usage, autosave=False)

                    # Pull token counts for the richer log entry (3.2)
                    prompt_tokens = getattr(response.usage, 'prompt_tokens', 0)
                    completion_tokens = getattr(response.usage, 'completion_tokens', 0)
                    tps = completion_tokens / latency if latency > 0 else 0

                    # 6. Score the Response
                    score_result = {"score": 0, "reason": "Scorer not matched"}

                    if test_case['category'] == "code_generation":
                        score_result = score_programmatic_code(target_text)
                    elif test_case['category'] == "metadata_generation":
                        score_result = score_programmatic_metadata(target_text, test_case['expected_schema_keys'])
                    elif test_case['category'] == "data_interpretation":
                        score_result = score_exact_match(target_text, test_case['expected_exact_match'])
                    elif test_case['category'] == "literature_synthesis":
                        # Guard against judge service missing from SERVICE_BUDGETS (1.2 fix)
                        judge_budget = SERVICE_BUDGETS.get(judge_service, _DEFAULT_BUDGET)
                        if tracker.check_budget(judge_service, judge_budget):
                            score_result = score_with_judge(
                                target_response=target_text,
                                prompt=test_case['prompt'],
                                rubric=test_case['judge_rubric'],
                                judge_client=judge_client,
                                judge_model=judge_model,
                                tracker=tracker
                            )
                            if "judge_usage" in score_result:
                                tracker.add_usage(judge_service, judge_model, score_result.pop("judge_usage"), autosave=False)
                        else:
                            score_result = {"score": 0, "reason": "Judge budget exceeded. Evaluation skipped."}

                    # Build richer log entry (3.2)
                    log_entry = {
                        "service": target_service_name,
                        "model": model_id,
                        "test_id": test_case['id'],
                        "category": test_case['category'],
                        "latency_sec": round(latency, 2),
                        "tps": round(tps, 2),
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "score": score_result["score"],
                        "reason": score_result["reason"],
                    }
                    # Preserve optional code output field when present
                    if "output" in score_result:
                        log_entry["code_output"] = score_result["output"]

                    results_log.append(log_entry)
                    completed_keys.add(run_key)
                    new_results_count += 1

                    logging.info(f"      -> Score: {score_result['score']}/5 | TPS: {round(tps, 2)}")

                except Exception as e:
                    logging.error(f"      -> Error testing {model_id} on {test_case['id']}: {e}")

            # Flush accumulated token usage to disk once per model (not once per task)
            tracker.save()

            if test_run and evaluated_count >= 2:
                logging.info(f"  Reached max 2 models for test run on {target_service_name}")
                break

    # 7. Save Output — always write back the full merged list (append behaviour)
    logging.info(f"\nRun complete. {new_results_count} new result(s) recorded.")
    try:
        with open(results_log_path, "w") as f:
            json.dump(results_log, f, indent=2)
        logging.info(f"Results saved to {results_log_path} ({len(results_log)} total entries)")
    except Exception as e:
        logging.error(f"Failed to save results: {e}")
