# Vald8 Documentation

Welcome to Vald8 - pytest for LLMs.

## Quick Links

- [Installation](installation.md)
- [Quick Start](quickstart.md)
- [Configuration](configuration.md)
- [Examples](examples.md)
- [API Reference](api.md)

## What is Vald8?

Vald8 is a minimalist, developer-first SDK for testing LLM-powered Python functions using structured JSONL datasets.

```python
from vald8 import vald8

@vald8(dataset="tests.jsonl")
def generate(prompt: str) -> dict:
    # Your LLM function
    return {"response": "..."}
```

## Features

- **Simple Decorator**: Just add `@vald8` to any function
- **Structured Testing**: Use JSONL datasets with expected outputs
- **Multiple Metrics**: Accuracy, schema validation, safety, custom judges
- **LLM-as-Judge**: Built-in support for OpenAI, Anthropic, Bedrock
- **Session Grouping**: Organize results by test run
- **Rich Reports**: JSON, JSONL, and human-readable text reports
