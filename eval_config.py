import os
from config import SERVICES

# Which services are we targeting for evaluation
TARGET_SERVICES = ["grit", "CIT"]

# ---------------------------------------------------------------------------
# LLM-as-a-Judge configuration
# ---------------------------------------------------------------------------
# Default judge: gemini-3.5-flash on dream-lab.  Cost-effective for iterative
# development runs.  For final / production benchmarks switch to a stronger
# frontier model via CLI flags:
#
#   uv run python run_eval.py --judge-service aicommons --judge-model claude-v4.6-sonnet
#
# Rule of thumb: the judge should be at least as capable as the strongest
# model under evaluation.  Using a weak judge introduces its own quality noise
# that inflates scores for verbose but inaccurate outputs.
DEFAULT_JUDGE_SERVICE = "dream-lab"
DEFAULT_JUDGE_MODEL = "gemini-3.5-flash"

# Recommended production judge (uncomment to change the default):
# DEFAULT_JUDGE_SERVICE = "aicommons"
# DEFAULT_JUDGE_MODEL   = "claude-v4.6-sonnet"

# ---------------------------------------------------------------------------
# Token / budget limits
# ---------------------------------------------------------------------------
# Global safety limits — the engine will stop calling a service once its
# monthly token count (tracked in .token_usage.json) reaches this threshold.
# Services not listed here fall back to _DEFAULT_BUDGET in eval_engine.py.
SERVICE_BUDGETS = {
    "aicommons": 900_000,   # strict 1 M/month quota; leave a 100 k safety buffer
    "dream-lab": 5_000_000,
    "grit":      10_000_000,
    "CIT":       10_000_000,
}

# The path to persist token usage across runs
TOKEN_TRACKER_FILE = ".token_usage.json"

# ---------------------------------------------------------------------------
# Model exclusion list
# ---------------------------------------------------------------------------
# Models to skip by default — embeddings, rerankers, OCR pipelines, or any
# model that is currently returning auth / capability errors.
# Use --force-all-models at the CLI to bypass this list.
EXCLUDED_MODELS = [
    "glm-ocr:latest",
    "bge-reranker",
    "qwen3-embedding-8b",
    "deepseek-v4-flash:cloud",
    "glm-5.2:cloud",
    "gemini-embedding-2",
]
