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

# 📁 JSONL Test Dataset Example

Save as `tests.jsonl`:

```json
{"id": "math1", "input": "What is 2+2?", "expected": {"reference": "4"}}
{"id": "json1", "input": "Return JSON with name and age", "expected": {"schema": {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "number"}}, "required": ["name", "age"]}}}
{"id": "hello1", "input": "Greet politely", "expected": {"contains": ["hello", "please"]}}
{"id": "regex1", "input": "Give a date", "expected": {"regex": "\d{4}-\d{2}-\d{2}" }}
```

Supported expectations:

- `"reference": "exact value"`
- `"contains": ["word1", "word2"]`
- `"regex": "pattern"`
- `"schema": {...}`  

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

# 🧩 CI/CD Integration

```yaml
- name: Run Vald8 Tests
  run: |
    python -c "
    from my_llm import generate
    assert generate.run_eval()['passed']
    "
```

---

# 📁 Results Format

Each run produces:

```
runs/
└── 2025-11-21_12-01-44/
    ├── results.jsonl
    ├── summary.json
    └── metadata.json
```

---

# 🔧 Configuration Options

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

# 🛠 Minimal Feature Set (v0.1)

Included:

- ✔ Test decorator  
- ✔ JSONL dataset loader  
- ✔ Schema validation  
- ✔ Contains / reference / regex checks  
- ✔ Optional LLM-as-judge  
- ✔ Clear results + artifacts  
- ✔ Offline mode  
- ✔ CI/CD-ready  
- ✔ Zero-config defaults  

---

# 🤝 Contributing

PRs welcome.

---

# 📜 License

MIT License — free and open source.
