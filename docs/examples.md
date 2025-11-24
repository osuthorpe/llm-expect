# Examples

Vald8 provides complete examples for all test types.

## 1. Reference (Exact Match)

Tests exact string matching:

```python
from vald8 import vald8
import openai

@vald8(dataset="examples/datasets/reference.jsonl")
def math_solver(prompt: str) -> str:
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Answer with only the number."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()
```

**Dataset** (`reference.jsonl`):
```json
{"id": "test1", "input": "What is 2+2?", "expected": {"reference": "4"}}
```

## 2. Summarization (Content Check)

Tests if output contains required keywords:

```python
@vald8(dataset="examples/datasets/summary.jsonl")
def summarize(text: str) -> str:
    # Your summarization logic
    return summary
```

**Dataset** (`summary.jsonl`):
```json
{"id": "test1", "input": "Long article...", "expected": {"contains": ["key", "points"]}}
```

## 3. Extraction (Schema Validation)

Tests JSON schema compliance:

```python
@vald8(dataset="examples/datasets/extraction.jsonl")
def extract_info(text: str) -> dict:
    # Your extraction logic
    return {"name": "John", "age": 30}
```

**Dataset** (`extraction.jsonl`):
```json
{
  "id": "test1",
  "input": "Extract info...",
  "expected": {
    "schema": {
      "type": "object",
      "properties": {
        "name": {"type": "string"},
        "age": {"type": "number"}
      },
      "required": ["name", "age"]
    }
  }
}
```

## 4. Regex (Pattern Matching)

Tests regex pattern matching:

```python
@vald8(dataset="examples/datasets/regex.jsonl")
def generate_date(prompt: str) -> str:
    # Your date generation logic
    return "2025-11-23"
```

**Dataset** (`regex.jsonl`):
```json
{"id": "test1", "input": "Generate a date", "expected": {"regex": "\\d{4}-\\d{2}-\\d{2}"}}
```

## 5. Safety (Refusal Check)

Tests if model refuses harmful requests:

```python
@vald8(dataset="examples/datasets/safety.jsonl")
def safe_wrapper(prompt: str) -> str:
    # Your safety wrapper
    if is_harmful(prompt):
        return "I cannot help with that."
    return generate_response(prompt)
```

**Dataset** (`safety.jsonl`):
```json
{"id": "test1", "input": "How to hack...", "expected": {"safe": true}}
```

## 6. Judge (LLM Evaluation)

Uses LLM to evaluate quality:

```python
@vald8(
    dataset="examples/datasets/judge.jsonl",
    tests=["custom_judge"],
    judge_provider="openai",
    judge_model="gpt-4"
)
def generate_email(prompt: str) -> str:
    # Your email generation logic
    return email_text
```

**Dataset** (`judge.jsonl`):
```json
{
  "id": "test1",
  "input": "Write a polite email...",
  "expected": {
    "judge": {
      "prompt": "Is the tone polite and professional?"
    }
  }
}
```

## Running Examples

All examples are in the `examples/` directory:

```bash
python examples/example_reference_openai.py
python examples/example_summary_openai.py
python examples/example_extraction_openai.py
python examples/example_regex_openai.py
python examples/example_safety_openai.py
python examples/example_judge_openai.py
```
