# 🧪 LLM Expect — Lightweight Evaluation Framework for LLM Reliability

LLM Expect is a minimalist, developer-first SDK for testing LLM-powered Python functions using structured JSONL datasets.

> 🤖 **For AI Assistants:** Read [`LLM_INSTRUCTIONS.md`](LLM_INSTRUCTIONS.md) for implementation patterns.

It provides a simple way to validate:

- **Schema correctness**
- **Instruction adherence**
- **Reference accuracy**
- **Keyword / regex expectations**

With optional support for **LLM-as-Judge** scoring.

Focus: **Make LLM evaluation as easy as pytest. Nothing more. Nothing less.**

---

# Why LLM Expect?

If you're building with LLMs, you need a way to verify that your AI functions:

- produce valid JSON  
- follow instructions consistently  
- don't regress when prompts or models change  
- behave consistently across environments  
- meet quality thresholds before deployment  

LLM Expect gives you this with:

- ✔ One decorator  
- ✔ One JSONL file  
- ✔ One evaluation call  

No configuration. No complexity. No over-engineering.

---

# Install

```bash
pip install llm-expect
```

---

# Core Concept

You decorate any LLM function:

```python
from llm_expect import llm_expect

@llm_expect(dataset="tests.jsonl")
def generate(prompt: str) -> dict:
    ...
```

LLM Expect loads your dataset, runs the function against each example, and scores the results.

---

## Running Examples

LLM Expect comes with a realistic example script that demonstrates how to evaluate functions using real LLM APIs (OpenAI, Anthropic, Gemini).

### Prerequisites

1.  **Install SDKs**:
    ```bash
    pip install openai anthropic google-generativeai
    ```

2.  **Set API Keys**:
    ```bash
    export OPENAI_API_KEY="your-key-here"
    export ANTHROPIC_API_KEY="your-key-here"
    export GEMINI_API_KEY="your-key-here"
    ```

    **Alternatively, use a `.env` file:**
    1. Copy `.env.example` to `.env`:
       ```bash
       cp .env.example .env
       ```
    2. Edit `.env` and add your API keys.

### Run the Examples

We provide specific examples for different testing scenarios:

**1. Reference (Exact Match)**
```bash
python examples/example_reference_openai.py
```

**2. Summarization (Content Check)**
```bash
python examples/example_summary_openai.py
```

**3. Extraction (Schema Validation)**
```bash
python examples/example_extraction_openai.py
```

**4. Regex (Pattern Matching)**
```bash
python examples/example_regex_openai.py
```

**5. Safety (Refusal Check)**
```bash
python examples/example_safety_openai.py
```

**6. Judge (LLM Evaluation)**
```bash
python examples/example_judge_openai.py
```

1.  Load the evaluation dataset from `examples/eval_dataset.jsonl`.
2.  Run evaluations on the respective models.
3.  Output pass/fail results and success rates.

### 🖥️ Rich CLI

LLM Expect includes a beautiful CLI for managing results:

```bash
# List recent runs
llm-expect runs list

# Show detailed results for a run
llm-expect runs show runs/2025-11-25_...
```

---

# 📁 JSONL Test Dataset Example

Save as `tests.jsonl`:

```json
{"id": "math1", "input": "What is 2+2?", "expected": {"reference": "4"}}
{"id": "json1", "input": "Return JSON with name and age", "expected": {"schema": {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "number"}}, "required": ["name", "age"]}}}
{"id": "hello1", "input": "Greet politely", "expected": {"contains": ["hello", "please"]}}
{"id": "regex1", "input": "Give a date", "expected": {"regex": "\d{4}-\d{2}-\d{2}" }}
```

Supported expectations:

### 1. Reference (Exact Match)
Checks if the output matches a reference string exactly (ignoring whitespace).
```json
"expected": {"reference": "42"}
```

### 2. Contains (Keywords)
Checks if the output contains **all** specified keywords (case-insensitive).
```json
"expected": {"contains": ["hello", "world"]}
```

### 3. Regex (Pattern Matching)
Checks if the output matches a regular expression.
```json
"expected": {"regex": "^\\d{4}-\\d{2}-\\d{2}$"}
```

### 4. Schema (JSON Validation)
Validates that the output is valid JSON and conforms to a JSON Schema.
```json
"expected": {
  "schema": {
    "type": "object",
    "properties": {
      "name": {"type": "string"},
      "age": {"type": "integer"}
    },
    "required": ["name", "age"]
  }
}
```

### 5. Safety (Harmful Content)
Checks for harmful content using a keyword list. Can be inverted.
```json
"expected": {"safe": true}
```

### 6. Judge (LLM-as-a-Judge)
Uses an LLM to evaluate the output based on a custom prompt. Requires `judge_provider` to be configured.
```json
"expected": {
  "judge": {
    "prompt": "Is this response polite and professional?"
  }
}
```


---

## ⚙️ Configuration

LLM Expect supports configuration through decorator parameters and environment variables.

### Decorator Parameters

```python
@llm_expect(
    dataset="path/to/dataset.jsonl",       # Required: Path to JSONL dataset
    tests=["accuracy", "schema_fidelity"], # Optional: Metrics to evaluate (default: [])
    thresholds={"accuracy": 0.9},          # Optional: Pass/fail thresholds (default: 0.8)
    judge_provider="openai",               # Optional: LLM judge provider
    judge_model="gpt-5.1",                 # Optional: Judge model name
    sample_size=10,                        # Optional: Number of examples to sample
    shuffle=True,                          # Optional: Shuffle before sampling (default: False)
    cache=True,                            # Optional: Cache results (default: True)
    cache_dir=".llm_expect_cache",              # Optional: Cache directory
    results_dir="runs",                    # Optional: Results directory
    fail_fast=False,                       # Optional: Stop on first failure (default: False)
    timeout=60,                            # Optional: Function timeout in seconds
    save_results=True,                     # Optional: Save detailed results (default: True)
    parallel=False                         # Optional: Run tests in parallel (default: False)
)
```

### Environment Variables

All configuration parameters can be set via environment variables with the `LLM_EXPECT_` prefix:

| Variable | Type | Description | Default |
|----------|------|-------------|---------|
| `LLM_EXPECT_TESTS` | List | Comma-separated metrics (e.g., `"accuracy,safety"`) | `[]` |
| `LLM_EXPECT_THRESHOLD` | Float | Global threshold for all metrics | `0.8` |
| `LLM_EXPECT_THRESHOLD_ACCURACY` | Float | Threshold for accuracy metric | `0.8` |
| `LLM_EXPECT_THRESHOLD_SAFETY` | Float | Threshold for safety metric | `1.0` |
| `LLM_EXPECT_SAMPLE_SIZE` | Int | Number of examples to sample | All |
| `LLM_EXPECT_SHUFFLE` | Bool | Shuffle examples (`true`/`false`) | `false` |
| `LLM_EXPECT_CACHE` | Bool | Enable caching | `true` |
| `LLM_EXPECT_CACHE_DIR` | String | Cache directory path | `.llm_expect_cache` |
| `LLM_EXPECT_RESULTS_DIR` | String | Results directory path | `runs` |
| `LLM_EXPECT_FAIL_FAST` | Bool | Stop on first failure | `false` |
| `LLM_EXPECT_TIMEOUT` | Int | Function timeout (seconds) | `60` |

### Judge Configuration

For LLM-as-judge metrics (`instruction_adherence`, `safety`, `custom_judge`):

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_EXPECT_JUDGE_MODEL` | Judge model name | Provider-specific |
| `LLM_EXPECT_JUDGE_API_KEY` | Judge API key | From provider env var |
| `LLM_EXPECT_JUDGE_BASE_URL` | Custom API base URL | Provider default |
| `LLM_EXPECT_JUDGE_TIMEOUT` | Judge request timeout | `30` |
| `LLM_EXPECT_JUDGE_MAX_RETRIES` | Max retry attempts | `3` |
| `LLM_EXPECT_JUDGE_TEMPERATURE` | Judge temperature | `0.0` |

**Provider API Keys:**
- OpenAI: `OPENAI_API_KEY`
- Anthropic: `ANTHROPIC_API_KEY`
- Bedrock: `AWS_ACCESS_KEY_ID`

---

# 🧪 Decorating an LLM Function

```python
from llm_expect import llm_expect
import openai

@llm_expect(dataset="tests.jsonl")
def generate(prompt: str) -> dict:
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return {"response": response.choices[0].message.content}
```

---

# 📊 Running Evaluations

```python
results = generate.run_eval()

print("Passed:", results["passed"])
print("Success Rate:", results["summary"]["success_rate"])
print("Details saved to:", results["run_dir"])
```

Example output:

```
✔ math1
✔ json1
✖ hello1 — missing: please
✔ regex1

Overall: 3/4 passed (75%)
```

---

# 🧱 Optional: LLM-as-Judge Scoring

Useful for long-form or fuzzy outputs.

```python
@llm_expect(
    dataset="tests.jsonl",
    tests=["schema", "contains", "reference"],
    thresholds={"success_rate": 0.9},
    sample_size=None,
    cache=False,
    judge_provider=None,
    parallel=True,
)
def summarize(text: str) -> str:
    return llm_summarize(text)
```

Most tests require **no API calls**.

---

# CI/CD Integration

```yaml
- name: Run LLM Expect Tests
  run: |
    python -c "
    from my_llm import generate
    assert generate.run_eval()['passed']
    "
```

---

# Results Format

Each run produces:

```
runs/
└── 2025-11-21_12-01-44/
    ├── results.jsonl
    ├── summary.json
    └── metadata.json
```

---

# ⚙️ Configuration Reference

## Decorator Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `dataset` | `str` | **Required** | Path to JSONL file (relative or absolute). |
| `tests` | `list[str]` | `[]` | Metrics to evaluate: `["accuracy", "schema_fidelity", "safety", "custom_judge"]`. |
| `thresholds` | `dict` | `{"accuracy": 0.8}` | Pass/fail thresholds per metric. |
| `judge_provider` | `str` | `None` | LLM judge provider: `"openai"`, `"anthropic"`, `"bedrock"`. |
| `judge_model` | `str` | Provider default | Specific model for the judge (e.g., `"gpt-4"`). |
| `sample_size` | `int` | `None` (All) | Number of examples to sample from the dataset. |
| `shuffle` | `bool` | `False` | Whether to shuffle examples before sampling. |
| `cache` | `bool` | `True` | Enable caching of results to avoid re-running passed tests. |
| `cache_dir` | `str` | `".llm_expect_cache"` | Directory for cache files. |
| `results_dir` | `str` | `"runs"` | Directory to save detailed evaluation results. |
| `fail_fast` | `bool` | `False` | Stop evaluation immediately on the first failure. |
| `timeout` | `int` | `60` | Timeout in seconds for the decorated function execution. |

## Environment Variables

All configuration can be overridden by environment variables with the `LLM_EXPECT_` prefix.

| Variable | Type | Description |
|----------|------|-------------|
| `LLM_EXPECT_TESTS` | List | Comma-separated metrics (e.g., `"accuracy,safety"`). |
| `LLM_EXPECT_THRESHOLD` | Float | Global threshold applied to all metrics. |
| `LLM_EXPECT_THRESHOLD_{METRIC}` | Float | Specific threshold for a metric (e.g., `LLM_EXPECT_THRESHOLD_SAFETY`). |
| `LLM_EXPECT_SAMPLE_SIZE` | Int | Number of examples to sample. |
| `LLM_EXPECT_SHUFFLE` | Bool | Shuffle examples (`true`/`false`). |
| `LLM_EXPECT_CACHE` | Bool | Enable/disable caching. |
| `LLM_EXPECT_CACHE_DIR` | Str | Cache directory path. |
| `LLM_EXPECT_RESULTS_DIR` | Str | Results directory path. |
| `LLM_EXPECT_FAIL_FAST` | Bool | Stop on first failure. |
| `LLM_EXPECT_TIMEOUT` | Int | Function timeout in seconds. |

## Judge Configuration

For metrics that require an LLM judge (`instruction_adherence`, `safety`, `custom_judge`):

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_EXPECT_JUDGE_MODEL` | Judge model name | Provider default (e.g., GPT-4) |
| `LLM_EXPECT_JUDGE_API_KEY` | Judge API key | `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` |
| `LLM_EXPECT_JUDGE_BASE_URL` | Custom API base URL | Provider default |
| `LLM_EXPECT_JUDGE_TIMEOUT` | Judge request timeout | `30` |
| `LLM_EXPECT_JUDGE_MAX_RETRIES` | Max retry attempts | `3` |
| `LLM_EXPECT_JUDGE_TEMPERATURE` | Judge temperature | `0.0` |
---

## 📁 Results Folder Structure

LLM Expect automatically saves evaluation results in a session-based hierarchy:

```
runs/
└── 2025-11-23_a1b2c3d4/              # Session folder (date + session_id)
    ├── extract_correct/              # Function-specific results
    │   ├── results.jsonl             # Detailed test results (one per line)
    │   ├── summary.json              # Aggregated statistics
    │   ├── metadata.json             # Run configuration and info
    │   └── report.txt                # Human-readable report
    ├── extract_incorrect/
    │   └── ...
    └── judge_correct/
        └── ...
```

**Session Grouping**: All functions evaluated in the same script run share a session ID and are grouped under one master folder.

**Files**:
- `results.jsonl`: Line-delimited JSON with each test result
- `summary.json`: Success rate, metrics, timing stats
- `metadata.json`: Function name, config, timestamp
- `report.txt`: Formatted text report
- `report.html`: Visual HTML report with charts and tables

---

# Contributing

PRs welcome.

---

# License

MIT License — free and open source.
