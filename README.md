# 🧪 Vald8 — Lightweight Evaluation Framework for LLM Reliability

Vald8 is a minimalist, developer-first SDK for testing LLM-powered Python functions using structured JSONL datasets.

It provides a simple way to validate:

- **Schema correctness**
- **Instruction adherence**
- **Reference accuracy**
- **Keyword / regex expectations**

With optional support for **LLM-as-Judge** scoring.

Focus: **Make LLM evaluation as easy as pytest. Nothing more. Nothing less.**

---

# Why Vald8?

If you're building with LLMs, you need a way to verify that your AI functions:

- produce valid JSON  
- follow instructions consistently  
- don't regress when prompts or models change  
- behave consistently across environments  
- meet quality thresholds before deployment  

Vald8 gives you this with:

- ✔ One decorator  
- ✔ One JSONL file  
- ✔ One evaluation call  

No configuration. No complexity. No over-engineering.

---

# Install

```bash
pip install vald8
```

---

# Core Concept

You decorate any LLM function:

```python
from vald8 import vald8

@vald8(dataset="tests.jsonl")
def generate(prompt: str) -> dict:
    ...
```

Vald8 loads your dataset, runs the function against each example, and scores the results.

---

## Running Examples

Vald8 comes with a realistic example script that demonstrates how to evaluate functions using real LLM APIs (OpenAI, Anthropic, Gemini).

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

**1. Summarization (Content Check)**
```bash
python examples/example_summary_openai.py
```

**2. Extraction (Schema Validation)**
```bash
python examples/example_extraction_openai.py
```

**3. Safety (Refusal Check)**
```bash
python examples/example_safety_openai.py
```

**4. Judge (LLM Evaluation)**
```bash
python examples/example_judge_openai.py
```

These scripts will:
1.  Load the evaluation dataset from `examples/eval_dataset.jsonl`.
2.  Run evaluations on the respective models.
3.  Output pass/fail results and success rates.

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

Vald8 supports configuration through decorator parameters and environment variables.

### Decorator Parameters

```python
@vald8(
    dataset="path/to/dataset.jsonl",       # Required: Path to JSONL dataset
    tests=["accuracy", "schema_fidelity"], # Optional: Metrics to evaluate (default: [])
    thresholds={"accuracy": 0.9},          # Optional: Pass/fail thresholds (default: 0.8)
    judge_provider="openai",               # Optional: LLM judge provider
    judge_model="gpt-5.1",                 # Optional: Judge model name
    sample_size=10,                        # Optional: Number of examples to sample
    shuffle=True,                          # Optional: Shuffle before sampling (default: False)
    cache=True,                            # Optional: Cache results (default: True)
    cache_dir=".vald8_cache",              # Optional: Cache directory
    results_dir="runs",                    # Optional: Results directory
    fail_fast=False,                       # Optional: Stop on first failure (default: False)
    timeout=60,                            # Optional: Function timeout in seconds
    save_results=True,                     # Optional: Save detailed results (default: True)
    parallel=False                         # Optional: Parallel execution (future)
)
```

### Environment Variables

All configuration parameters can be set via environment variables with the `VALD8_` prefix:

| Variable | Type | Description | Default |
|----------|------|-------------|---------|
| `VALD8_TESTS` | List | Comma-separated metrics (e.g., `"accuracy,safety"`) | `[]` |
| `VALD8_THRESHOLD` | Float | Global threshold for all metrics | `0.8` |
| `VALD8_THRESHOLD_ACCURACY` | Float | Threshold for accuracy metric | `0.8` |
| `VALD8_THRESHOLD_SAFETY` | Float | Threshold for safety metric | `1.0` |
| `VALD8_SAMPLE_SIZE` | Int | Number of examples to sample | All |
| `VALD8_SHUFFLE` | Bool | Shuffle examples (`true`/`false`) | `false` |
| `VALD8_CACHE` | Bool | Enable caching | `true` |
| `VALD8_CACHE_DIR` | String | Cache directory path | `.vald8_cache` |
| `VALD8_RESULTS_DIR` | String | Results directory path | `runs` |
| `VALD8_FAIL_FAST` | Bool | Stop on first failure | `false` |
| `VALD8_TIMEOUT` | Int | Function timeout (seconds) | `60` |

### Judge Configuration

For LLM-as-judge metrics (`instruction_adherence`, `safety`, `custom_judge`):

| Variable | Description | Default |
|----------|-------------|---------|
| `VALD8_JUDGE_MODEL` | Judge model name | Provider-specific |
| `VALD8_JUDGE_API_KEY` | Judge API key | From provider env var |
| `VALD8_JUDGE_BASE_URL` | Custom API base URL | Provider default |
| `VALD8_JUDGE_TIMEOUT` | Judge request timeout | `30` |
| `VALD8_JUDGE_MAX_RETRIES` | Max retry attempts | `3` |
| `VALD8_JUDGE_TEMPERATURE` | Judge temperature | `0.0` |

**Provider API Keys:**
- OpenAI: `OPENAI_API_KEY`
- Anthropic: `ANTHROPIC_API_KEY`
- Bedrock: `AWS_ACCESS_KEY_ID`

---

# 🧪 Decorating an LLM Function

```python
from vald8 import vald8
import openai

@vald8(dataset="tests.jsonl")
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
@vald8(
    dataset="tests.jsonl",
    judge_provider="openai"   # or "anthropic", "local"
)
def summarize(text: str) -> str:
    return llm_summarize(text)
```

Most tests require **no API calls**.

---

# CI/CD Integration

```yaml
- name: Run Vald8 Tests
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

# Configuration Options

```python
@vald8(
    dataset="tests.jsonl",
    tests=["schema", "contains", "reference"],
    thresholds={"success_rate": 0.9},
    sample_size=None,
    cache=False,
    judge_provider=None,
)
```

All parameters are optional.

---

## 📁 Results Folder Structure

Vald8 automatically saves evaluation results in a session-based hierarchy:

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
- `report.txt`: Formatted report with failed tests

---

# Contributing

PRs welcome.

---

# License

MIT License — free and open source.
