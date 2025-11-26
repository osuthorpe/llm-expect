# Installation

## Requirements

- Python 3.9 or higher
- pip

## Install from PyPI

```bash
pip install llm-expect
```

## Install from Source

```bash
git clone https://github.com/osuthorpe/llm-expect.git
cd llm-expect
pip install -e .
```

## Verify Installation

```python
import llm_expect
print(llm_expect.__version__)
```

## Optional Dependencies

For development:

```bash
pip install "llm-expect[dev]"
```

For documentation:

```bash
pip install "llm-expect[docs]"
```

For everything:

```bash
pip install llm-expect[all]
```
