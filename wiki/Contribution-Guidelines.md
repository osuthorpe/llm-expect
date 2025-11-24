# Contribution Guidelines

We are thrilled that you are interested in contributing to Vald8! We are building a community of developers who care about **simple, transparent, and rigorous AI evaluation**.

## Our Values

1.  **Be Kind**: We are all learning. Constructive feedback is welcome; toxicity is not.
2.  **Be Pragmatic**: We value working code over theoretical purity.
3.  **Be Transparent**: Discuss major changes in an issue before writing code.

## How to Contribute

### 1. Reporting Bugs
Found a bug? Please open an issue on GitHub. Include:
*   Steps to reproduce.
*   Expected behavior vs. actual behavior.
*   Your environment (OS, Python version).

### 2. Suggesting Features
Have an idea? We'd love to hear it! Open an issue with the tag `enhancement`.
*   Explain the problem you are solving.
*   Propose a solution (API design, implementation).
*   Check the [Roadmap](Roadmap.md) to see if it aligns with our direction.

### 3. Submitting Pull Requests (PRs)
1.  **Fork the repo** and create your branch from `main`.
2.  **Install dependencies**: `pip install -r requirements.txt` and `pip install -e .`
3.  **Run tests**: Make sure all existing tests pass with `pytest`.
4.  **Add tests**: If you are adding a feature, add a test case for it.
5.  **Lint your code**: We use standard Python formatting (PEP 8).
6.  **Submit the PR**: Link the issue you are addressing.

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/vald8-sdk.git
cd vald8-sdk

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in editable mode
pip install -e .

# Run tests
pytest
```

## License
By contributing, you agree that your contributions will be licensed under the MIT License.
