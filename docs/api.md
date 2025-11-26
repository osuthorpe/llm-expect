# API Reference

This reference documents the core components of the `llm-expect` library.

## Core Decorator

The main entry point is the `@llm_expect` decorator.

### Usage Example

```python
from llm_expect import llm_expect

@llm_expect(
    dataset="tests.jsonl",
    tests=["accuracy", "safety"],
    thresholds={"accuracy": 0.9}
)
def my_llm_function(prompt: str) -> str:
    # ... implementation ...
    return "response"

# Run the evaluation
results = my_llm_function.run_eval()
```

::: llm_expect.decorator
    options:
      show_root_heading: true
      show_source: true
      members:
        - llm_expect

## Data Models

These models define the structure of inputs and results.

::: llm_expect.models
    options:
      show_root_heading: true
      members:
        - EvaluationResult
        - EvaluationSummary
        - TestResult
        - MetricResult
        - LLMExpectConfig

## Configuration

::: llm_expect.config
    options:
      show_root_heading: true
      members:
        - LLMExpectConfig
