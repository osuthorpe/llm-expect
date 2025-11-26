# Contributing

Thank you for your interest in contributing to LLM Expect!

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/llm-expect.git
cd llm-expect
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install development dependencies:
```bash
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/ -v
```

## Code Style

We use:
- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

Run all checks:
```bash
black vald8 tests
isort vald8 tests
flake8 vald8 tests
mypy vald8
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Run code formatters and linters
7. Commit your changes (`git commit -m 'Add amazing feature'`)
8. Push to your fork (`git push origin feature/amazing-feature`)
9. Open a Pull Request

## Reporting Issues

Please use the [GitHub issue tracker](https://github.com/osuthorpe/vald8/issues) to report bugs or request features.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
