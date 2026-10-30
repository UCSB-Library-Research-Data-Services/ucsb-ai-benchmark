import argparse
import json
import csv
import os
from eval_engine import run_evaluation, re_judge_from_log
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

    # Group by service + model + category + judge_model
    # Including judge_model lets the CSV show separate rows for each judge,
    # enabling direct score comparison across judges in the dashboard.
    summary = {}
    for row in data:
        if row.get("score") is None or row.get("judge_conflict"):
            continue   # skip conflict-skipped entries from the summary

        judge = row.get("judge_model", "programmatic")
        key = f"{row['service']}::{row['model']}::{row['category']}::{judge}"
        if key not in summary:
            summary[key] = {
                "service":        row["service"],
                "model":          row["model"],
                "category":       row["category"],
                "judge_model":    judge,
                "total_tests":    0,
                "total_score":    0,
                "total_tps":      0.0,
                "tps_samples":    0,
                "total_latency":  0.0,
                "latency_samples":0,
            }

        s = summary[key]
        s["total_tests"]  += 1
        s["total_score"]  += row["score"]
        if row.get("tps", 0) > 0:
            s["total_tps"]    += row["tps"]
            s["tps_samples"]  += 1
        if row.get("latency_sec", 0) > 0:
            s["total_latency"]    += row["latency_sec"]
            s["latency_samples"]  += 1

    with open(csv_path, "w", newline="") as csvfile:
        fieldnames = [
            "service", "model", "category", "judge_model",
            "tests_run", "avg_score", "avg_latency_sec", "avg_tps",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for s in summary.values():
            avg_score   = s["total_score"]   / s["total_tests"]    if s["total_tests"]    else 0
            avg_tps     = s["total_tps"]     / s["tps_samples"]    if s["tps_samples"]    else 0
            avg_latency = s["total_latency"] / s["latency_samples"] if s["latency_samples"] else 0
            writer.writerow({
                "service":          s["service"],
                "model":            s["model"],
                "category":         s["category"],
                "judge_model":      s["judge_model"],
                "tests_run":        s["total_tests"],
                "avg_score":        round(avg_score, 2),
                "avg_latency_sec":  round(avg_latency, 2),
                "avg_tps":          round(avg_tps, 2),
            })
    print(f"Summary saved to {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run LLM Evaluations for UCSB Services",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available datasets:
  evaluations/eval_dataset.jsonl            Standard v1 benchmark (text only)
  evaluations/eval_dataset_v2.jsonl         Harder v2 benchmark (text only)
  evaluations/eval_dataset_multimodal.jsonl OCR, document extraction, and vision QA

Available services:  grit, CIT, dream-lab, aicommons

Modes:
  Normal evaluation  (default)
  Re-judge mode      --re-judge  (re-scores stored model outputs with a new judge)

Examples:
  # Normal run
  uv run python run_eval.py --dataset evaluations/eval_dataset_v2.jsonl --services grit CIT

  # Re-judge all gemini-scored literature_synthesis entries with claude
  uv run python run_eval.py --re-judge \\
      --judge-service aicommons --judge-model claude-v4.6-sonnet \\
      --source-judge gemini-3.5-flash \\
      --include-judge-detail

  # Re-judge everything (all judges, all categories)
  uv run python run_eval.py --re-judge \\
      --judge-service aicommons --judge-model claude-v4.6-sonnet
        """,
    )

    # ── Mode flag ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--re-judge",
        action="store_true",
        help=(
            "Re-judge mode: re-score stored model outputs in results_log.json "
            "with a new judge, without calling target models again. "
            "Results are stored alongside originals — originals are never overwritten."
        ),
    )

    # ── Re-judge filters (only used with --re-judge) ───────────────────────
    parser.add_argument(
        "--source-judge",
        type=str,
        default=None,
        metavar="JUDGE_MODEL",
        help=(
            "Re-judge mode only: restrict to entries originally scored by this judge "
            "(e.g. 'gemini-3.5-flash'). Omit to re-judge entries from all judges."
        ),
    )
    parser.add_argument(
        "--re-judge-categories",
        type=str,
        nargs="+",
        default=None,
        metavar="CATEGORY",
        help=(
            "Re-judge mode only: restrict to these categories "
            "(e.g. --re-judge-categories literature_synthesis ocr). "
            "Default: all judge-scored categories."
        ),
    )

    # ── Shared flags ───────────────────────────────────────────────────────
    parser.add_argument(
        "--dataset",
        type=str, default="evaluations/eval_dataset.jsonl",
        help="Dataset file for normal evaluation (default: evaluations/eval_dataset.jsonl). Ignored in --re-judge mode.",
    )
    parser.add_argument(
        "--services",
        type=str, nargs="+", default=None, metavar="SERVICE",
        help=(
            f"Services to evaluate / filter re-judge entries by "
            f"(default: {TARGET_SERVICES}). "
            "Example: --services grit CIT"
        ),
    )
    parser.add_argument("--test-run", action="store_true",
                        help="Run only the first task against max 2 models (normal mode only).")
    parser.add_argument("--judge-service", type=str, default=DEFAULT_JUDGE_SERVICE,
                        help=f"Service for the Judge LLM (default: {DEFAULT_JUDGE_SERVICE}).")
    parser.add_argument("--judge-model", type=str, default=DEFAULT_JUDGE_MODEL,
                        help=f"Model for the Judge LLM (default: {DEFAULT_JUDGE_MODEL}).")
    parser.add_argument("--force-all-models", action="store_true",
                        help="Bypass the exclusion list (normal mode only).")
    parser.add_argument("--include-model-output", action="store_true",
                        help="Store full model responses in results_log.json.")
    parser.add_argument("--include-judge-detail", action="store_true",
                        help="Include full judge reasoning (requires --include-model-output in normal mode).")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-recorded triplets (normal mode; re-judge mode always resumes).")
    parser.add_argument(
        "--allow-self-judge", action="store_true",
        help="Disable the judge-conflict guard. NOT recommended for production.",
    )

    args = parser.parse_args()

    # ── Shared validation ──────────────────────────────────────────────────
    known_services = {"grit", "CIT", "aicommons"}
    target_services = args.services or TARGET_SERVICES
    unknown = [s for s in target_services if s not in known_services]
    if unknown:
        parser.error(f"Unknown service(s): {unknown}. Available: {sorted(known_services)}")

    if not args.re_judge and args.include_judge_detail and not args.include_model_output:
        parser.error("--include-judge-detail requires --include-model-output")

    # ── RE-JUDGE mode ──────────────────────────────────────────────────────
    if args.re_judge:
        print("=======================================")
        print("Re-Judge Mode")
        print(f"  New judge  : {args.judge_service} / {args.judge_model}")
        print(f"  Source judge filter: {args.source_judge or 'all'}")
        print(f"  Categories : {args.re_judge_categories or 'all judge-scored'}")
        print(f"  Services   : {target_services}")
        print("=======================================")

        re_judge_from_log(
            judge_service=args.judge_service,
            judge_model=args.judge_model,
            source_judge=args.source_judge,
            categories=args.re_judge_categories,
            services=list(target_services),
            include_judge_detail=args.include_judge_detail,
            allow_self_judge=args.allow_self_judge,
        )

        print("\n=======================================")
        print("Generating Summary Report...")
        print("=======================================")
        generate_summary_csv()
        return

    # ── NORMAL evaluation mode ─────────────────────────────────────────────
    if not os.path.exists(args.dataset):
        parser.error(f"Dataset file not found: '{args.dataset}'")

    print("=======================================")
    print("Starting Evaluation Pipeline...")
    print(f"  Dataset  : {args.dataset}")
    print(f"  Services : {list(target_services)}")
    print(f"  Judge    : {args.judge_service} / {args.judge_model}")
    print("=======================================")
    run_evaluation(
        dataset_path=args.dataset,
        target_services=list(target_services),
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
