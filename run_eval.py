import argparse
import json
import csv
import os
from eval_engine import run_evaluation
from eval_config import DEFAULT_JUDGE_SERVICE, DEFAULT_JUDGE_MODEL

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
                "avg_tps": 0.0,
                "tps_samples": 0
            }
        
        summary[key]['total_tests'] += 1
        summary[key]['total_score'] += row['score']
        if row.get('tps', 0) > 0:
            summary[key]['avg_tps'] += row['tps']
            summary[key]['tps_samples'] += 1

    # Write CSV
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['service', 'model', 'category', 'tests_run', 'avg_score', 'avg_tps']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for key, stats in summary.items():
            avg_score = stats['total_score'] / stats['total_tests'] if stats['total_tests'] > 0 else 0
            avg_tps = stats['avg_tps'] / stats['tps_samples'] if stats['tps_samples'] > 0 else 0
            
            writer.writerow({
                'service': stats['service'],
                'model': stats['model'],
                'category': stats['category'],
                'tests_run': stats['total_tests'],
                'avg_score': round(avg_score, 2),
                'avg_tps': round(avg_tps, 2)
            })
    print(f"Summary saved to {csv_path}")

def main():
    parser = argparse.ArgumentParser(description="Run LLM Evaluations for UCSB Services")
    parser.add_argument("--test-run", action="store_true", help="Run a quick test with 1 task and limited models.")
    parser.add_argument("--judge-service", type=str, default=DEFAULT_JUDGE_SERVICE, help=f"Service to use for the Judge LLM (default: {DEFAULT_JUDGE_SERVICE}).")
    parser.add_argument("--judge-model", type=str, default=DEFAULT_JUDGE_MODEL, help=f"Model to use for the Judge LLM (default: {DEFAULT_JUDGE_MODEL}).")
    
    args = parser.parse_args()
    
    print("=======================================")
    print("Starting Evaluation Pipeline...")
    print("=======================================")
    run_evaluation(test_run=args.test_run, judge_service=args.judge_service, judge_model=args.judge_model)
    
    print("\n=======================================")
    print("Generating Summary Report...")
    print("=======================================")
    generate_summary_csv()

if __name__ == "__main__":
    main()
