# UCSB AI Services Evaluation Suite

This project provides an automated evaluation pipeline to benchmark Large Language Models (LLMs) provided across various UCSB AI services (e.g., `grit`, `CIT`, `aicommons`, `dream-lab`). 

It is specifically designed to evaluate models on their utility for **scientific research workflows** rather than generic conversational capabilities.

## Architecture & Logic

The evaluation suite tests two primary dimensions:
1. **Inference Performance:** Standardized metrics like latency and Tokens Per Second (TPS).
2. **Scientific Intelligence:** Domain-specific capabilities evaluated through programmatic validation and "LLM-as-a-Judge" scoring.

### Evaluation Categories
The test cases (defined in `eval_dataset.json`) fall into four categories, each with a specific scoring mechanism (`eval_scorers.py`):

1. **Code Generation (Data Science):** 
   - *Logic:* The model is asked to write a Python script (e.g., using `pandas` to process data). The code is extracted, injected with dummy data, and executed in an isolated subprocess.
   - *Score:* 5/5 if it executes cleanly, 1/5 if it throws an error.
2. **Metadata Generation (Data Deposits):**
   - *Logic:* The model generates structured metadata (e.g., DataCite) from raw text. The output is parsed to ensure it is valid JSON and contains required schema keys.
   - *Score:* 5/5 for perfect JSON, 2/5 for missing keys, 0/5 for invalid JSON.
3. **Literature Review & Synthesis:**
   - *Logic:* The model is asked to extract methodologies and conclusions from scientific abstracts.
   - *Score (LLM Judge):* A highly capable "Frontier" model (like Claude 4.6 or Gemini 3.5) acts as a Judge. It reads the target model's output and scores it 1-5 based on a strict rubric for accuracy and lack of hallucination.
4. **Data Interpretation:**
   - *Logic:* The model answers multiple-choice questions based on statistical summaries.
   - *Score:* 5/5 for exact match, 1/5 for incorrect.

### Token & Budget Management
Because LLM-as-a-Judge can quickly consume API quotas (especially on `aicommons` which has a strict 1M tokens/month limit), the suite includes a `TokenTracker` (`token_tracker.py`). It tracks token usage from API responses and enforces hard monthly limits to prevent budget overruns.

## Installation

Ensure you have `uv` and `python` installed. 
You must set up your API keys in a `.env` file (see `config.py` for required keys like `GRIT_KEY`, `CIT_KEY`, etc.).

```bash
# Add necessary dependencies for programmatic code evaluation
uv add pandas numpy
```

## How to Run

The main entry point is `run_eval.py`.

**1. Run a quick test (Recommended first step)**
Runs only the first task in the dataset against a maximum of 2 models per service to ensure the pipeline is working.
```bash
uv run python run_eval.py --test-run
```

**2. Run the full benchmark suite (Default)**
Evaluates all active models on all tasks. By default, it uses `dream-lab` (`gemini-3.5-flash`) as the LLM Judge to save costs.
```bash
uv run python run_eval.py
```

**3. Run with a specific LLM Judge ("Production" Benchmark)**
For final evaluations, you can swap the Judge to a more capable model available on `aicommons`.
```bash
uv run python run_eval.py --judge-service aicommons --judge-model claude-v4.6-sonnet
```

**4. Force run all models**
By default, the script skips known problematic models (e.g., embeddings, rerankers, models returning 400 errors) listed in `eval_config.py`. To bypass this and evaluate all available models:
```bash
uv run python run_eval.py --force-all-models
```

## Advanced Features

### Robustness & Resume

Results are written to `results_log.json` **after each task** to ensure durability. If your run is interrupted (connection drop, rate limit, etc.), simply resume:

```bash
uv run python run_eval.py --resume
```

The engine automatically skips any `(service, model, test_id)` triplet already in `results_log.json` and continues where it left off. No data is lost!

### Transparency & Detailed Logging

By default, results only include scores and brief reasons. For deeper inspection and debugging:

#### Include Full Model Outputs
```bash
uv run python run_eval.py --include-model-output
```

Adds `model_output` and `output_type` fields to every result entry in `results_log.json`. Useful for:
- Debugging low scores by seeing the actual model response
- Auditing model behavior across different services
- Building detailed dashboards with side-by-side comparisons

Example output in `results_log.json`:
```json
{
  "service": "grit",
  "model": "gemma4:31b",
  "test_id": "code_gen_1",
  "category": "code_generation",
  "score": 5,
  "reason": "Execution successful with numeric output",
  "model_output": "import pandas as pd\ndf = pd.read_csv('data.csv')\nfiltered = df[df['temperature'] > 20]\nmean_pressure = filtered['pressure'].mean()\nprint(mean_pressure)",
  "output_type": "code"
}
```

#### Include Judge Reasoning (Literature Synthesis Only)
```bash
uv run python run_eval.py --include-model-output --include-judge-detail
```

Adds detailed judge reasoning and rubric feedback to judge-based evaluations. Requires `--include-model-output`.

**Note:** This increases token usage for judge calls (requests multi-sentence feedback), so only use when you need it.

Example for literature_synthesis:
```json
{
  "service": "grit",
  "model": "gemma4:31b",
  "test_id": "lit_synthesis_1",
  "category": "literature_synthesis",
  "score": 4,
  "reason": "Accurate extraction with minor detail missing",
  "model_output": "{\"methodology\": \"GPS tracking and direct observation\", \"species\": \"sea otter (Enhydra lutris)\", \"conclusion\": \"top-down control aids kelp restoration\"}",
  "output_type": "json",
  "judge_reasoning": "The model correctly identified GPS tracking and direct observation as the methodology. The species identification is accurate. The conclusion captures the main finding. However, the connection to 'targeted' control mechanism was slightly underemphasized.",
  "judge_rubric_feedback": "See detailed feedback above.",
  "judge_model": "gemini-3.5-flash"
}
```

### Example: Production Benchmark with Full Transparency

```bash
uv run python run_eval.py \
  --judge-service aicommons \
  --judge-model claude-v4.6-sonnet \
  --include-model-output \
  --include-judge-detail
```

This produces a fully auditable results log suitable for stakeholder review and dashboard generation.

## Reading the Results

The pipeline generates two files upon completion:

### 1. `results_summary.csv` (High-Level Scorecard)
This provides aggregated averages, making it easy to see which model is best at specific tasks.

| service | model | category | tests_run | avg_score | avg_tps |
| :--- | :--- | :--- | :--- | :--- | :--- |
| CIT | gemma-4-31b | code_generation | 1 | 5.0 | 695.48 |
| CIT | gemma-4-31b | literature_synthesis | 1 | 4.0 | 500.12 |
| grit | gemma4:31b | code_generation | 1 | 1.0 | 8.89 |

- **`avg_score`**: Average score out of 5.0. Higher is "smarter".
- **`avg_tps`**: Average Tokens Per Second. Higher is "faster".

### 2. `results_log.json` (Deep Dive / Debugging)

This file contains the raw execution logs. If a model scored poorly in the summary, check this file to find out *why*.

**Baseline entry (always present):**
```json
{
  "service": "grit",
  "model": "gemma4:31b",
  "test_id": "code_gen_1",
  "category": "code_generation",
  "latency_sec": 80.14,
  "tps": 3.38,
  "prompt_tokens": 450,
  "completion_tokens": 1532,
  "score": 1,
  "reason": "Execution failed: ModuleNotFoundError: No module named 'pandas'"
}
```

The `reason` field explicitly shows why the model failed the code generation task, allowing you to debug whether the issue was the model's logic or a missing dependency in the testing environment.

**Optional fields (with `--include-model-output`):**
- `model_output` — The full response from the target model (code, JSON, text, etc.)
- `output_type` — Category of output for dashboard rendering (`code`, `json`, `multiple_choice`, `text`)

**Optional fields (with `--include-judge-detail`, literature_synthesis only):**
- `judge_reasoning` — Full multi-sentence feedback from the judge
- `judge_rubric_feedback` — Detailed scoring rationale
- `judge_model` — Which judge model scored this entry

