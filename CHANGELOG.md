# Changelog

All notable changes to **LLM Expect** (formerly Vald8) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.9] - 2025-11-26

### Changed
- **Lazy Initialization**: Refactored `EvaluationRunner` to initialize judge providers and metrics only when `run_eval()` is called. This prevents `ImportError` or `JudgeProviderError` during test collection if API keys are missing.
- **Documentation**: Added "Pytest Integration & CI" section to `README.md` and `llm.txt` with best practices for handling import-time side effects in CI environments.

## [0.1.8] - 2025-11-26

### Added
- **Documentation**: Added `docs/dataset_examples.md` with robust JSONL examples.
- **LLM Accessibility**: Added `docs/llm.txt` and `docs/llms.txt` to make instructions easily accessible to AI agents.
- **Examples**: Added comprehensive examples to `llm.txt` for all test types.

### Changed
- **Documentation**: Updated `README.md` with detailed configuration, installation, and usage instructions.
- **Documentation**: Updated `llm.txt` with clearer implementation patterns and metric inference rules.

## [0.1.7] - 2025-11-25

### 🚨 Breaking Changes
- **Project Rename**: Renamed project from `vald8` to `llm-expect`.
    - Package name: `llm-expect`
    - Import: `import llm_expect`
    - Decorator: `@llm_expect`
    - CLI: `llm-expect`
- **Configuration**: `Vald8Config` renamed to `LLMExpectConfig`.
- **Errors**: `Vald8Error` renamed to `LLMExpectError`.

### Added
- `llm.txt`: Comprehensive context file for LLMs.

## [0.1.6] - 2025-11-25

### Added
- **Parallel Execution**: Added `parallel=True` argument to `@vald8` decorator to run tests concurrently using `ThreadPoolExecutor`.
- **Rich CLI**: Updated CLI to use `rich` library for beautiful, tabular output of evaluation runs.
- **HTML Reports**: Automatic generation of `report.html` for every evaluation run, featuring charts and detailed test results.
- Support for decorating instance methods directly with `@vald8`. The decorator now implements the descriptor protocol to correctly bind `self`.
- New dependencies: `rich` and `jinja2`.

### Changed
- Updated documentation to reflect that module-level wrappers are no longer required for class methods.
- Refactored examples to demonstrate direct class method decoration.

## [0.1.5] - 2025-11-24

### Added
- Class method examples and documentation.

### Changed
- Updated Pydantic model serialization to V2 syntax.

## [0.1.4] - 2025-11-24

### Changed
- Migrated Pydantic model configuration to V2 syntax.

## [0.1.3] - 2025-11-24

### Added
- GitHub Actions workflows for testing and PyPI publishing.
- Initial project wiki documentation.

## [0.1.2] - 2025-11-24

### Fixed
- Updated PyPI links in configuration.

## [0.1.1] - 2025-11-23

### Added
- Initial release of Vald8 SDK.