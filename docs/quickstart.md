# Quickstart

## 1. Create a Dataset

Create a JSONL file with test cases (`tests.jsonl`):

```json
{"id": "test1", "input": "What is 2+2?", "expected": {"reference": "4"}}
{"id": "test2", "input": "Return JSON with name", "expected": {"schema": {"type": "object", "properties": {"name": {"type": "string"}}}}}
```

## 2. Decorate Your Function

```python
from llm_expect import llm_expect
import openai

@llm_expect(dataset="tests.jsonl")
def generate(prompt: str) -> str:
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
```

## 3. Run Evaluation

```python
# Run the evaluation
results = generate.run_eval()

# Check results
print(f"Passed: {results['passed']}")
print(f"Success Rate: {results['summary']['success_rate']:.1%}")
```

## 4. View Results

Results are automatically saved to `runs/`:

```
runs/
└── 2025-11-23_a1b2c3d4/
    └── generate/
        ├── results.jsonl
        ├── summary.json
        ├── metadata.json
        └── report.txt
```

## Next Steps

- Learn about [Configuration](configuration.md)
- Explore [Examples](examples.md)
- Read the [API Reference](api.md)
