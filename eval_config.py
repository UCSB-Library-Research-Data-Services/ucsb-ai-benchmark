import os
from config import SERVICES

# Which services are we targeting for evaluation
TARGET_SERVICES = ["grit", "CIT"]

# Default settings for the LLM-as-a-Judge
DEFAULT_JUDGE_SERVICE = "dream-lab"
DEFAULT_JUDGE_MODEL = "gemini-3.5-flash"

# Global safety limits (e.g., aicommons is strictly limited to 1M per month)
SERVICE_BUDGETS = {
    "aicommons": 900_000, # Leave a 100k safety buffer
    "dream-lab": 5_000_000, # Assuming a generous quota for testing
    "grit": 10_000_000,
    "CIT": 10_000_000,
}

# The path to persist token usage across runs
TOKEN_TRACKER_FILE = ".token_usage.json"

# Models to exclude by default (e.g., embeddings, rerankers, OCR, or currently unauthorized models)
EXCLUDED_MODELS = [
    "glm-ocr:latest",
    "bge-reranker",
    "qwen3-embedding-8b",
    "deepseek-v4-flash:cloud",
    "glm-5.2:cloud",
    "gemini-embedding-2"
]
