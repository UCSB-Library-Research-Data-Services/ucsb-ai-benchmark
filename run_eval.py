import argparse
import json
import csv
import os
from eval_engine import run_evaluation
from eval_config import DEFAULT_JUDGE_SERVICE, DEFAULT_JUDGE_MODEL, TARGET_SERVICES

def generate_summary_csv(json_path="results_log.json", csv_path="results_summary.csv"):
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found. Cannot generate summary.")
        return

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return

    if not data:
        print("No data to summarize.")
        return

    # Group by service + model + category
    summary = {}
    for row in data:
        key = f"{row['service']}::{row['model']}::{row['category']}"
        if key not in summary:
            summary[key] = {
                "service": row['service'],
                "model": row['model'],
                "category": row['category'],
                "total_tests": 0,
                "total_score": 0,
                "total_tps": 0.0,
                "tps_samples": 0,
                "total_latency": 0.0,
                "latency_samples": 0,
            }

        summary[key]['total_tests'] += 1
        summary[key]['total_score'] += row['score']
        if row.get('tps', 0) > 0:
            summary[key]['total_tps'] += row['tps']
            summary[key]['tps_samples'] += 1
        if row.get('latency_sec', 0) > 0:
            summary[key]['total_latency'] += row['latency_sec']
            summary[key]['latency_samples'] += 1

    # Write CSV
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['service', 'model', 'category', 'tests_run', 'avg_score', 'avg_latency_sec', 'avg_tps']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for key, stats in summary.items():
            avg_score = stats['total_score'] / stats['total_tests'] if stats['total_tests'] > 0 else 0
            avg_tps = stats['total_tps'] / stats['tps_samples'] if stats['tps_samples'] > 0 else 0
            avg_latency = stats['total_latency'] / stats['latency_samples'] if stats['latency_samples'] > 0 else 0

            writer.writerow({
                'service': stats['service'],
                'model': stats['model'],
                'category': stats['category'],
                'tests_run': stats['total_tests'],
                'avg_score': round(avg_score, 2),
                'avg_latency_sec': round(avg_latency, 2),
                'avg_tps': round(avg_tps, 2),
            })
    print(f"Summary saved to {csv_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Run LLM Evaluations for UCSB Services",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available datasets:
  eval_dataset.json            Standard v1 benchmark (text only)
  eval_dataset_v2.json         Harder v2 benchmark (text only)
  eval_dataset_multimodal.json OCR, document extraction, and vision QA (requires vision-capable models)

Available services:
  grit, CIT, dream-lab, aicommons

Examples:
  uv run python run_eval.py --dataset eval_dataset_v2.json
  uv run python run_eval.py --dataset eval_dataset_multimodal.json --services dream-lab aicommons --include-model-output
  uv run python run_eval.py --dataset eval_dataset_v2.json --resume --include-model-output --include-judge-detail
        """,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="eval_dataset.json",
        help="Path to the evaluation dataset JSON file (default: eval_dataset.json).",
    )
    parser.add_argument(
        "--services",
        type=str,
        nargs="+",
        default=None,
        metavar="SERVICE",
        help=(
            f"One or more services to evaluate (default: {TARGET_SERVICES}). "
            "Available: grit, CIT, dream-lab, aicommons. "
            "Example: --services dream-lab aicommons"
        ),
    )
    parser.add_argument("--test-run", action="store_true", help="Run only the first task against a maximum of 2 models per service.")
    parser.add_argument("--judge-service", type=str, default=DEFAULT_JUDGE_SERVICE, help=f"Service for the Judge LLM (default: {DEFAULT_JUDGE_SERVICE}).")
    parser.add_argument("--judge-model", type=str, default=DEFAULT_JUDGE_MODEL, help=f"Model for the Judge LLM (default: {DEFAULT_JUDGE_MODEL}).")
    parser.add_argument("--force-all-models", action="store_true", help="Bypass the exclusion list and test all active models.")
    parser.add_argument("--include-model-output", action="store_true", help="Include full target model responses in results_log.json.")
    parser.add_argument("--include-judge-detail", action="store_true", help="Include full judge reasoning and rubric feedback (requires --include-model-output).")
    parser.add_argument("--resume", action="store_true", help="Skip (service, model, test_id) triplets already recorded in results_log.json.")
    parser.add_argument(
        "--allow-self-judge",
        action="store_true",
        help=(
            "Disable the judge-conflict guard and allow a model to be evaluated "
            "by a judge from the same model family. NOT recommended for production benchmarks — "
            "use only for debugging or when you explicitly want self-comparison."
        ),
    )

    args = parser.parse_args()

    # Validate flag dependencies
    if args.include_judge_detail and not args.include_model_output:
        parser.error("--include-judge-detail requires --include-model-output")

    if not os.path.exists(args.dataset):
        parser.error(f"Dataset file not found: '{args.dataset}'")

    # Validate --services values
    known_services = {"grit", "CIT", "dream-lab", "aicommons"}
    target_services = args.services or TARGET_SERVICES
    unknown = [s for s in target_services if s not in known_services]
    if unknown:
        parser.error(f"Unknown service(s): {unknown}. Available: {sorted(known_services)}")

    print("=======================================")
    print(f"Starting Evaluation Pipeline...")
    print(f"  Dataset  : {args.dataset}")
    print(f"  Services : {target_services}")
    print(f"  Judge    : {args.judge_service} / {args.judge_model}")
    print("=======================================")
    run_evaluation(
        dataset_path=args.dataset,
        target_services=target_services,
        test_run=args.test_run,
        judge_service=args.judge_service,
        judge_model=args.judge_model,
        force_all_models=args.force_all_models,
        include_model_output=args.include_model_output,
        include_judge_detail=args.include_judge_detail,
        resume=args.resume,
        allow_self_judge=args.allow_self_judge,
    )
    
    print("\n=======================================")
    print("Generating Summary Report...")
    print("=======================================")
    generate_summary_csv()

if __name__ == "__main__":
    main()
