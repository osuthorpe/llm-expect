# LLM Expect Examples

This directory contains examples demonstrating various features and use cases of the LLM Expect evaluation framework.

## Quick Start

All examples require an OpenAI API key. Set it in your `.env` file:

```bash
OPENAI_API_KEY=your_api_key_here
```

Then run any example:

```bash
python examples/example_judge_openai.py
```

## Examples Overview

### Basic Examples

- **`example_judge_openai.py`** - Custom judge evaluation using OpenAI
- **`example_reference_openai.py`** - Reference-based evaluation
- **`example_regex_openai.py`** - Regex pattern matching evaluation
- **`example_safety_openai.py`** - Safety and content moderation evaluation
- **`example_summary_openai.py`** - Summary quality evaluation
- **`example_extraction_openai.py`** - Information extraction evaluation

### Advanced Examples

- **`example_class_method.py`** - Using `@llm_expect` with class methods ⭐

## Using LLM Expect with Class Methods

The `@llm_expect` decorator is designed to work with **standalone functions**, not instance methods. If you have a class with methods you want to evaluate, you need to create a module-level wrapper function.

### ❌ This Won't Work

```python
class MyClass:
    @vald8(dataset="data.jsonl")
    def my_method(self, input: str) -> str:  # ❌ Error: missing 'self'
        return self.process(input)
```

**Error:** `Function call failed: my_method() missing 1 required positional argument: 'self'`

### ✅ Solution 1: Module-Level Wrapper (Recommended)

Create a module-level function that delegates to your class method:

```python
class MyClass:
    def my_method(self, input: str) -> str:
        return self.process(input)

# Module-level instance
_instance = None

def _get_instance():
    global _instance
    if _instance is None:
        _instance = MyClass()
    return _instance

# Decorate the wrapper function
@vald8(dataset="data.jsonl")
def my_method(input: str) -> str:
    """Wrapper for LLM Expect evaluation."""
    instance = _get_instance()
    return instance.my_method(input)
```

See [`example_class_method.py`](./example_class_method.py) for a complete working example.

### ✅ Solution 2: Static Method

If your method doesn't need instance state:

```python
class MyClass:
    @vald8(dataset="data.jsonl")
    @staticmethod
    def my_method(input: str) -> str:
        return process(input)
```

### ✅ Solution 3: Class Method

If you need access to class-level state:

```python
class MyClass:
    config = {"model": "gpt-4"}
    
    @vald8(dataset="data.jsonl")
    @classmethod
    def my_method(cls, input: str) -> str:
        return process(input, model=cls.config["model"])
```

## Dataset Format

All examples use JSONL (JSON Lines) format for datasets. Each line is a JSON object:

```jsonl
{"id": "test_1", "input": "test input", "expected": {"judge": {"prompt": "Is this good?"}}}
{"id": "test_2", "input": {"field1": "value1", "field2": "value2"}, "expected": {"judge": {"prompt": "Is this correct?"}}}
```

### Input Formats

**Simple string input:**
```json
{"id": "test_1", "input": "Write a polite email", "expected": {...}}
```

**Structured input (multiple parameters):**
```json
{
  "id": "test_1",
  "input": {
    "title": "Fix bug",
    "description": "Fixed crash",
    "labels": ["urgent", "bug"]
  },
  "expected": {...}
}
```

When using structured input, your function signature must match the input fields:

```python
@vald8(dataset="data.jsonl")
def my_function(title: str, description: str, labels: List[str]) -> str:
    # Function parameters match the input object keys
    pass
```

## Running Examples

### Run a single example:
```bash
python examples/example_judge_openai.py
```

### Run with virtual environment:
```bash
source .venv/bin/activate
python examples/example_class_method.py
```

### View evaluation results:
Results are automatically displayed in the console. For detailed results, check the `.vald8_cache` directory.

## Common Issues

### Issue: "missing 1 required positional argument: 'self'"

**Cause:** You're trying to decorate an instance method directly.

**Solution:** Use one of the wrapper patterns shown in [`example_class_method.py`](./example_class_method.py).

### Issue: Dataset not found

**Cause:** Incorrect path to dataset file.

**Solution:** Use relative paths from the project root or absolute paths:
```python
@vald8(dataset="examples/datasets/my_data.jsonl")  # Relative to project root
```

### Issue: Input parameters don't match

**Cause:** Function signature doesn't match the dataset input structure.

**Solution:** Ensure your function parameters match the keys in the `input` object:

```json
// Dataset
{"input": {"title": "...", "description": "..."}}
```

```python
# Function must have matching parameters
def my_function(title: str, description: str) -> str:
    pass
```

## Need Help?

- 📖 [Full Documentation](https://vald8.readthedocs.io/)
- 🐛 [Report Issues](https://github.com/osuthorpe/Vald8-sdk/issues)
- 💬 [Discussions](https://github.com/osuthorpe/Vald8-sdk/discussions)
