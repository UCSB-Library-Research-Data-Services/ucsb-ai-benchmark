import json
import time
import logging
from config import SERVICES
from eval_config import TARGET_SERVICES, DEFAULT_JUDGE_SERVICE, DEFAULT_JUDGE_MODEL, SERVICE_BUDGETS
from eval_scorers import score_programmatic_code, score_programmatic_metadata, score_exact_match, score_with_judge
from token_tracker import TokenTracker
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def get_client(service_name: str) -> OpenAI:
    config = SERVICES.get(service_name)
    if not config or not config.get("key"):
        raise ValueError(f"Service {service_name} is missing or has no API key.")
    return OpenAI(
        base_url=config.get("url"),
        api_key=config.get("key"),
        timeout=120.0
    )

def run_evaluation(test_run: bool = False, judge_service: str = DEFAULT_JUDGE_SERVICE, judge_model: str = DEFAULT_JUDGE_MODEL):
    tracker = TokenTracker()
    
    # 1. Load Dataset
    try:
        with open("eval_dataset.json", "r") as f:
            dataset = json.load(f)
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        return

    if test_run:
        dataset = dataset[:1] # Just run the first test
        logging.info("TEST RUN MODE: Only running 1 test case.")

    # 2. Initialize Judge Client
    logging.info(f"Initializing Judge Client: {judge_service} / {judge_model}")
    try:
        judge_client = get_client(judge_service)
    except Exception as e:
        logging.error(f"Failed to initialize judge: {e}")
        return

    results_log = []

    # 3. Iterate Target Services
    for target_service_name in TARGET_SERVICES:
        logging.info(f"\n--- Evaluating Service: {target_service_name} ---")
        
        # Check target service budget
        if not tracker.check_budget(target_service_name, SERVICE_BUDGETS[target_service_name]):
            logging.warning(f"Skipping {target_service_name} due to budget limits.")
            continue

        try:
            target_client = get_client(target_service_name)
            # Fetch active models for this service
            models_page = target_client.models.list()
            active_models = [m.id for m in models_page.data]
        except Exception as e:
            logging.error(f"Failed to connect to {target_service_name}: {e}")
            continue

        if test_run:
            # Evaluate max 2 models per service on test run, ensuring we skip failing/cloud models for testing
            active_models = [m for m in active_models if "cloud" not in m and "embedding" not in m and "reranker" not in m][:2]

        # 4. Iterate Models & Prompts
        for model_id in active_models:
            # Filter out known embedding/reranker models before evaluation
            if "embedding" in model_id.lower() or "reranker" in model_id.lower():
                continue
            
            logging.info(f"  Testing Model: {model_id}")
            
            for test_case in dataset:
                logging.info(f"    Task: {test_case['category']} ({test_case['id']})")
                
                start_time = time.time()
                try:
                    # Call the Target Model
                    response = target_client.chat.completions.create(
                        model=model_id,
                        messages=[{"role": "user", "content": test_case['prompt']}]
                    )
                    latency = time.time() - start_time
                    
                    target_text = response.choices[0].message.content
                    
                    # Track target token usage
                    tracker.add_usage(target_service_name, model_id, response.usage)
                    
                    # Calculate TPS
                    completion_tokens = getattr(response.usage, 'completion_tokens', 0)
                    tps = completion_tokens / latency if latency > 0 else 0

                    # 5. Score the Response
                    score_result = {"score": 0, "reason": "Scorer not matched"}
                    
                    if test_case['category'] == "code_generation":
                        score_result = score_programmatic_code(target_text)
                    elif test_case['category'] == "metadata_generation":
                        score_result = score_programmatic_metadata(target_text, test_case['expected_schema_keys'])
                    elif test_case['category'] == "data_interpretation":
                        score_result = score_exact_match(target_text, test_case['expected_exact_match'])
                    elif test_case['category'] == "literature_synthesis":
                        # Check Judge Budget before calling
                        if tracker.check_budget(judge_service, SERVICE_BUDGETS[judge_service]):
                            score_result = score_with_judge(
                                target_response=target_text,
                                prompt=test_case['prompt'],
                                rubric=test_case['judge_rubric'],
                                judge_client=judge_client,
                                judge_model=judge_model,
                                tracker=tracker
                            )
                            # Record judge usage if present
                            if "judge_usage" in score_result:
                                tracker.add_usage(judge_service, judge_model, score_result.pop("judge_usage"))
                        else:
                            score_result = {"score": 0, "reason": "Judge budget exceeded. Evaluation skipped."}

                    # Append to results
                    results_log.append({
                        "service": target_service_name,
                        "model": model_id,
                        "test_id": test_case['id'],
                        "category": test_case['category'],
                        "latency_sec": round(latency, 2),
                        "tps": round(tps, 2),
                        "score": score_result["score"],
                        "reason": score_result["reason"]
                    })
                    logging.info(f"      -> Score: {score_result['score']}/5 | TPS: {round(tps,2)}")

                except Exception as e:
                    logging.error(f"      -> Error testing {model_id} on {test_case['id']}: {e}")

    # 6. Save Output
    try:
        with open("results_log.json", "w") as f:
            json.dump(results_log, f, indent=2)
        logging.info("\nEvaluation complete. Results saved to results_log.json")
    except Exception as e:
        logging.error(f"Failed to save results: {e}")


