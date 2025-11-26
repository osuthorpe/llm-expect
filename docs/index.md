# LLM Expect Documentation

Welcome to LLM Expect - pytest for LLMs.

## Quick Links

- [Installation](installation.md)
- **[Quickstart](quickstart.md)**: Get started with LLM Expect in minutes.
- [Examples](examples.md)
- [API Reference](api.md)

## Welcome to the documentation for **LLM Expect**, a minimalist Python SDK for evaluating LLM functions.

LLM Expect is a minimalist, developer-first SDK for testing LLM-powered Python functions using structured JSONL datasets.

```python
from llm_expect import llm_expect

@llm_expect(dataset="tests.jsonl")
def generate(prompt: str) -> dict:
    # Your LLM function
    return {"response": "..."}
```

## Features

- **Simple Decorator**: Just add `@llm_expect` to any function
- **Structured Testing**: Use JSONL datasets with expected outputs
- **Multiple Metrics**: Accuracy, schema validation, safety, custom judges
- **LLM-as-Judge**: Built-in support for OpenAI, Anthropic, Bedrock
- **Session Grouping**: Organize results by test run
- **Rich Reports**: JSON, JSONL, and human-readable text reports
