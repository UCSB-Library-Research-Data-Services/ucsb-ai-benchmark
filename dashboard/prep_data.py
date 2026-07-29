"""
prep_data.py
Filters dream-lab out of results_summary.csv and results_log.json,
writes cleaned versions to dashboard/data/ for the Quarto website.
Also computes a few derived fields used by the OJS visualisations.

Run from the project root:
    uv run python dashboard/prep_data.py
"""

import json
import csv
import pathlib
import re

ROOT = pathlib.Path(__file__).parent.parent
SRC_SUMMARY = ROOT / "results_summary.csv"
SRC_LOG     = ROOT / "results_log.json"
OUT_DIR     = ROOT / "dashboard" / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXCLUDE_SERVICE = {"dream-lab"}

# ---------------------------------------------------------------------------
# Infrastructure metadata injected into output
# ---------------------------------------------------------------------------
INFRA = {
    "grit":      {"label": "GRIT",      "infra": "On-premise · Ollama",  "infra_type": "onprem",  "url": "llm.grit.ucsb.edu"},
    "CIT":       {"label": "CIT",       "infra": "On-premise · Ollama",  "infra_type": "onprem",  "url": "api.ai.college.ucsb.edu"},
    "aicommons": {"label": "AICommons", "infra": "Cloud · Amazon Bedrock","infra_type": "cloud",   "url": "zkh52rh785.execute-api.us-east-1.amazonaws.com"},
}

# Canonical category ordering / display names
CATEGORY_LABELS = {
    "code_generation":     "Code Generation",
    "metadata_generation": "Metadata Generation",
    "literature_synthesis":"Literature Synthesis",
    "data_interpretation": "Data Interpretation",
    "ocr":                 "OCR",
    "document_extraction": "Document Extraction",
    "vision_qa":           "Vision QA",
}

# Cross-service model family groupings (for intersection chart)
MODEL_FAMILIES = {
    "Gemma 4 · 27B":        {"grit": "gemma4:31b",            "CIT": "gemma-4-31b"},
    "Qwen3-Coder-Next":     {"grit": "qwen3-coder-next:latest","CIT": "Qwen3-Coder-Next"},
    "GPT-OSS":              {"grit": "gpt-oss:20b",            "CIT": "gpt-oss-120b"},
    "Qwen3 / Qwen3.5":      {"grit": "qwen3.5:latest",         "CIT": "qwen3.6-35b-a3b"},
    "Qwen3 (base)":         {"grit": "qwen3:latest"},
    "Qwen3-Coder":          {"grit": "qwen3-coder:latest"},
    "Mistral":              {"grit": "mistral:latest",          "CIT": "mistral-small-4-119b-2603"},
    "LLaMA 3.1 · 8B":       {"grit": "llama3.1:8b"},
    "LLaMA 3":              {"grit": "llama3:latest"},
    "LLaVA · 7B":           {"grit": "llava:7b"},
    "Phi-4":                {"grit": "phi4:latest"},
    "Gemma 3":              {"grit": "gemma3:latest"},
    "DeepSeek-R1":          {"grit": "deepseek-r1:latest"},
    "All-The-Things":       {"grit": "all-the-things"},
    "Granite 4.1 · 30B":    {"CIT": "granite-4.1-30b"},
    "GPT-4o":               {"aicommons": "gpt-4o"},
}

# ---------------------------------------------------------------------------
# 1. Filter + enrich results_summary.csv
# ---------------------------------------------------------------------------
rows = []
with open(SRC_SUMMARY, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["service"] in EXCLUDE_SERVICE:
            continue
        svc = row["service"]
        row["service_label"]  = INFRA[svc]["label"]
        row["infra"]          = INFRA[svc]["infra"]
        row["infra_type"]     = INFRA[svc]["infra_type"]
        row["category_label"] = CATEGORY_LABELS.get(row["category"], row["category"])
        row["avg_score"]      = float(row["avg_score"])
        row["avg_latency_sec"]= float(row["avg_latency_sec"])
        row["avg_tps"]        = float(row["avg_tps"])
        row["tests_run"]      = int(row["tests_run"])
        rows.append(row)

fieldnames = list(rows[0].keys())
out_summary = OUT_DIR / "results_summary.csv"
with open(out_summary, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"  summary rows written: {len(rows)}  →  {out_summary}")

# ---------------------------------------------------------------------------
# 2. Filter results_log.json
# ---------------------------------------------------------------------------
with open(SRC_LOG) as f:
    log = json.load(f)

log_filtered = []
for entry in log:
    if entry.get("service") in EXCLUDE_SERVICE:
        continue
    svc = entry.get("service", "")
    entry["service_label"] = INFRA.get(svc, {}).get("label", svc)
    entry["infra_type"]    = INFRA.get(svc, {}).get("infra_type", "")
    entry["category_label"]= CATEGORY_LABELS.get(entry.get("category",""), entry.get("category",""))
    log_filtered.append(entry)

out_log = OUT_DIR / "results_log.json"
with open(out_log, "w") as f:
    json.dump(log_filtered, f, indent=2)

print(f"  log entries written:  {len(log_filtered)}  →  {out_log}")

# ---------------------------------------------------------------------------
# 3. Build model_families.json  (used by Overview intersection chart)
# ---------------------------------------------------------------------------
families_out = []
for family_name, services_map in MODEL_FAMILIES.items():
    families_out.append({
        "family":   family_name,
        "services": services_map,
        "service_list": sorted(services_map.keys()),
    })

out_families = OUT_DIR / "model_families.json"
with open(out_families, "w") as f:
    json.dump(families_out, f, indent=2)

print(f"  families written:     {len(families_out)}  →  {out_families}")

# ---------------------------------------------------------------------------
# 4. Build leaderboard.json  (overall avg score per service+model, canonical judge)
# ---------------------------------------------------------------------------
# For literature_synthesis, prefer claude-v4.6-sonnet; skip gemini rows when
# a claude row exists for the same (service, model, category).
from collections import defaultdict

# First pass: collect canonical rows
canonical = {}  # key = (service, model, category)
for row in rows:
    key = (row["service"], row["model"], row["category"])
    judge = row["judge_model"]
    existing = canonical.get(key)
    if existing is None:
        canonical[key] = row
    else:
        # prefer claude over gemini
        if judge == "claude-v4.6-sonnet":
            canonical[key] = row

# Second pass: aggregate per (service, model)
agg = defaultdict(lambda: {"scores": [], "tps_vals": [], "latency_vals": [], "categories": set()})
for (svc, mdl, cat), row in canonical.items():
    k = (svc, mdl)
    agg[k]["scores"].append(row["avg_score"])
    agg[k]["tps_vals"].append(row["avg_tps"])
    agg[k]["latency_vals"].append(row["avg_latency_sec"])
    agg[k]["categories"].add(cat)
    agg[k]["service"] = svc
    agg[k]["model"]   = mdl
    agg[k]["service_label"] = row["service_label"]
    agg[k]["infra"]   = row["infra"]
    agg[k]["infra_type"] = row["infra_type"]
    # per-category scores
    agg[k][f"score_{cat}"] = row["avg_score"]
    agg[k][f"tps_{cat}"]   = row["avg_tps"]

leaderboard = []
for (svc, mdl), data in agg.items():
    overall = round(sum(data["scores"]) / len(data["scores"]), 3)
    avg_tps  = round(sum(data["tps_vals"]) / len(data["tps_vals"]), 1)
    avg_lat  = round(sum(data["latency_vals"]) / len(data["latency_vals"]), 2)
    entry = {
        "service":       svc,
        "model":         mdl,
        "service_label": data["service_label"],
        "infra":         data["infra"],
        "infra_type":    data["infra_type"],
        "overall_score": overall,
        "avg_tps":       avg_tps,
        "avg_latency_sec": avg_lat,
        "categories_tested": len(data["categories"]),
    }
    for cat in CATEGORY_LABELS:
        entry[f"score_{cat}"] = data.get(f"score_{cat}", None)
    leaderboard.append(entry)

# Sort by overall score desc, then tps desc
leaderboard.sort(key=lambda x: (-x["overall_score"], -x["avg_tps"]))

out_lb = OUT_DIR / "leaderboard.json"
with open(out_lb, "w") as f:
    json.dump(leaderboard, f, indent=2)

print(f"  leaderboard entries:  {len(leaderboard)}  →  {out_lb}")

# ---------------------------------------------------------------------------
# 5. Build per_test_scores.json  (heatmap data: model × test_id → score)
#    canonical judge selection same as above
# ---------------------------------------------------------------------------
# key: (service, model, test_id) → best entry
test_canonical = {}
for entry in log_filtered:
    key = (entry["service"], entry["model"], entry["test_id"])
    judge = entry.get("judge_model", "programmatic")
    existing = test_canonical.get(key)
    if existing is None:
        test_canonical[key] = entry
    else:
        if judge == "claude-v4.6-sonnet":
            test_canonical[key] = entry

per_test = []
for (svc, mdl, tid), entry in test_canonical.items():
    per_test.append({
        "service":       svc,
        "model":         mdl,
        "service_label": INFRA.get(svc, {}).get("label", svc),
        "test_id":       tid,
        "category":      entry.get("category", ""),
        "category_label": CATEGORY_LABELS.get(entry.get("category",""), ""),
        "score":         entry.get("score", 0),
        "latency_sec":   entry.get("latency_sec", 0),
        "tps":           entry.get("tps", 0),
        "reason":        entry.get("reason", ""),
        "judge_model":   entry.get("judge_model", "programmatic"),
    })

out_pt = OUT_DIR / "per_test_scores.json"
with open(out_pt, "w") as f:
    json.dump(per_test, f, indent=2)

print(f"  per-test rows:        {len(per_test)}  →  {out_pt}")

# ---------------------------------------------------------------------------
# 6. Build judge_comparison.json  (lit_synthesis rows with both judges)
# ---------------------------------------------------------------------------
judge_rows = defaultdict(dict)  # (service, model, test_id) → {gemini: score, claude: score, ...}
for entry in log_filtered:
    if entry.get("category") != "literature_synthesis":
        continue
    key = (entry["service"], entry["model"], entry["test_id"])
    judge = entry.get("judge_model", "")
    judge_rows[key]["service"]  = entry["service"]
    judge_rows[key]["model"]    = entry["model"]
    judge_rows[key]["test_id"]  = entry["test_id"]
    judge_rows[key]["service_label"] = INFRA.get(entry["service"], {}).get("label", entry["service"])
    if "gemini" in judge:
        judge_rows[key]["score_gemini"] = entry["score"]
        judge_rows[key]["reason_gemini"] = entry.get("reason", "")
    elif "claude" in judge:
        judge_rows[key]["score_claude"] = entry["score"]
        judge_rows[key]["reason_claude"] = entry.get("reason", "")

judge_comparison = [v for v in judge_rows.values()
                    if "score_gemini" in v and "score_claude" in v]

out_jc = OUT_DIR / "judge_comparison.json"
with open(out_jc, "w") as f:
    json.dump(judge_comparison, f, indent=2)

print(f"  judge comparison rows:{len(judge_comparison)}  →  {out_jc}")

print("\nAll data files ready.")
