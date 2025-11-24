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

# 🚀 Why Vald8?

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

# 📦 Install

```bash
pip install vald8
```

---

# 🧩 Core Concept

You decorate any LLM function:

```python
from vald8 import vald8

@vald8(dataset="tests.jsonl")
def generate(prompt: str) -> dict:
    ...
```

Vald8 loads your dataset, runs the function against each example, and scores the results.

---

## 🚀 Running Examples

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

# Contributing

PRs welcome.

---

# License

MIT License — free and open source.
