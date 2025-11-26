# How to Extend LLM-Expect

While LLM Expect aims to be simple, it is also designed to be extensible. You can add custom metrics, judges, and integration patterns.

## Custom Metrics

Currently, `llm-expect` supports a fixed set of metrics (`accuracy`, `schema_fidelity`, `safety`, `instruction_adherence`, `custom_judge`).
However, you can implement custom logic **within your decorated function** or by pre-processing your dataset.

### Pattern: The "Wrapper" Function
If you need to check a complex condition that isn't covered by built-in metrics, wrap your LLM call:

```python
@llm_expect(dataset="tests.jsonl")
def generate_and_validate(prompt: str):
    response = call_llm(prompt)
    
    # Custom validation logic
    if "forbidden_term" in response:
        # You can modify the return value to trigger a failure in a 'contains' check
        return "FAILURE: Forbidden term detected"
        
    return response
```

## Custom Judges

You can use any LLM as a judge by configuring the `judge_provider`.
While we support OpenAI, Anthropic, and Bedrock out of the box, you can extend this by implementing the `JudgeProvider` abstract base class if you fork the library.

(Future versions will allow registering custom providers via entry points).

## Pytest Integration

As mentioned in the [Quickstart](index.md), you can integrate with `pytest`.
For advanced usage, you can create a custom pytest fixture that loads your JSONL data and dynamically generates test cases.

```python
# tests/test_dynamic.py
import pytest
import json
from my_app import generate

def load_tests():
    with open("tests.jsonl") as f:
        return [json.loads(line) for line in f]

@pytest.mark.parametrize("test_case", load_tests())
def test_llm_expect_logic(test_case):
    # This is a manual test, bypassing the @llm_expect runner
    # Useful if you want to use pytest's reporting instead
    result = generate(test_case["input"])
    # ... assert logic ...
```

## Agent Step Testing

LLM Expect is great for testing individual steps of an agent.
Instead of testing the whole agent loop, decorate the specific tool-use functions:

```python
@llm_expect(dataset="tool_tests.jsonl")
def decide_tool(user_input: str) -> dict:
    # Tests if the agent picks the right tool for a query
    return agent.decide(user_input)
```
