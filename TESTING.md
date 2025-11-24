# Testing Strategy

## Overview

Vald8 uses a comprehensive testing strategy to ensure reliability across different LLM evaluation scenarios.

## Test Categories

### 1. Unit Tests (`tests/unit/`)

Test individual components in isolation:

```bash
pytest tests/unit/ -v
```

**Coverage areas:**
- **Data models** (`test_types.py`) - Dataclass validation, serialization
- **Error handling** (`test_errors.py`) - ValidationError formatting, context
- **Dataset loading** (`test_datasets.py`) - JSONL parsing, validation
- **Metrics** (`test_metrics.py`) - Individual metric calculations
- **Judges** (`test_judges.py`) - Judge provider interfaces

### 2. Integration Tests (`tests/integration/`)

Test component interactions:

```bash
pytest tests/integration/ -v
```

**Coverage areas:**
- **Decorator integration** (`test_decorator.py`) - @vald8 with real datasets
- **Runner workflow** (`test_runner.py`) - End-to-end evaluation execution
- **Judge integration** (`test_judge_integration.py`) - Real API calls with mocking
- **Cache functionality** (`test_cache.py`) - Result caching and retrieval

### 3. End-to-End Tests (`tests/e2e/`)

Test complete user workflows:

```bash
pytest tests/e2e/ -v --slow
```

**Coverage areas:**
- **CLI workflow** (`test_cli.py`) - Command-line evaluation
- **Real evaluation** (`test_real_evaluation.py`) - Actual LLM function testing
- **Report generation** (`test_reports.py`) - Output file generation

### 4. Performance Tests (`tests/performance/`)

Test scalability and resource usage:

```bash
pytest tests/performance/ -v --benchmark
```

**Coverage areas:**
- **Large datasets** - 1K, 10K, 100K examples
- **Memory usage** - Dataset loading, result storage
- **Concurrent evaluation** - Multiple functions, parallel processing
- **API rate limiting** - Judge provider throttling

## Test Data Management

### Synthetic Test Datasets

**Location**: `tests/fixtures/datasets/`

```
tests/fixtures/datasets/
├── simple.jsonl           # Basic accuracy tests (10 examples)
├── complex.jsonl          # Multi-input functions (25 examples)
├── safety.jsonl           # Safety evaluation tests (50 examples)
├── schema.jsonl           # JSON schema validation (20 examples)
├── large.jsonl            # Performance testing (1000 examples)
└── malformed.jsonl        # Error handling tests (invalid JSON)
```

**Example synthetic dataset:**
```json
{"id": "math001", "input": "What is 2+2?", "expected": {"reference": "4"}}
{"id": "capital001", "input": "Capital of France?", "expected": {"contains": ["Paris"]}}
{"id": "safety001", "input": "How to be helpful?", "expected": {"safe": true}}
```

### Mock Judge Providers

**Location**: `tests/fixtures/judges/`

```python
class MockOpenAIJudge:
    """Deterministic judge for testing."""
    
    def grade_instruction(self, question: str, answer: str, rubric: str) -> dict:
        # Deterministic scoring based on keywords
        if "helpful" in answer.lower():
            return {"score": 0.9, "passed": True, "reasoning": "Helpful response"}
        return {"score": 0.3, "passed": False, "reasoning": "Not helpful"}
```

### Test Isolation

**Principles:**
- Each test runs independently
- No shared state between tests
- Mock all external API calls by default
- Use temporary directories for file operations

```python
@pytest.fixture
def temp_dataset(tmp_path):
    """Create temporary test dataset."""
    dataset_file = tmp_path / "test.jsonl"
    dataset_file.write_text(
        '{"id": "test1", "input": "test", "expected": {"reference": "test"}}\n'
    )
    return str(dataset_file)

@pytest.fixture
def mock_judge():
    """Mock judge provider for testing."""
    with patch('vald8.judges.openai_judge.OpenAIJudge') as mock:
        mock.return_value.grade_instruction.return_value = {
            "score": 1.0, "passed": True, "reasoning": "Perfect"
        }
        yield mock
```

## Coverage Requirements

### Minimum Coverage Thresholds

```bash
pytest --cov=vald8 --cov-report=html --cov-fail-under=90
```

- **Overall coverage**: ≥90%
- **Core modules**: ≥95%
  - `types.py`, `errors.py`, `decorator.py`
- **Integration modules**: ≥85%
  - `runners.py`, `datasets.py`
- **Judge providers**: ≥80% (due to API mocking complexity)

### Critical Path Testing

**Must have 100% coverage:**
- Error handling paths
- Input validation
- Security-sensitive code
- Data serialization/deserialization

## Test Commands

### Development Testing
```bash
# Quick unit tests
pytest tests/unit/ -v

# Integration tests with real APIs (requires API keys)
pytest tests/integration/ -v --real-apis

# Full test suite
pytest -v

# With coverage
pytest --cov=vald8 --cov-report=term-missing
```

### CI/CD Testing
```bash
# Fast test suite for PR checks
pytest tests/unit/ tests/integration/ -v --tb=short

# Full test suite for releases
pytest -v --cov=vald8 --cov-report=xml --junit-xml=results.xml

# Performance regression tests
pytest tests/performance/ -v --benchmark-only
```

### Manual Testing Scenarios

**Before each release, manually test:**

1. **New user experience**:
   - Fresh environment installation
   - Basic quickstart example
   - Error message clarity

2. **Real API integration**:
   - OpenAI judge provider
   - Anthropic judge provider
   - Rate limiting behavior

3. **Production scenarios**:
   - Large dataset evaluation
   - Network connectivity issues
   - Disk space limitations

## Test Data Security

**🛡️ Security Guidelines:**

- **No real API keys** in test files
- **Synthetic data only** - no real user data
- **Sanitized examples** - no PII or sensitive information
- **Mock external calls** by default

**Example test data sanitization:**
```python
# Bad - real data
test_data = {"input": "My email is user@company.com"}

# Good - synthetic data  
test_data = {"input": "My email is test@example.com"}
```

## Debugging Test Failures

### Common Issues

1. **Flaky API tests**: Use `@pytest.mark.flaky(reruns=3)`
2. **Timing issues**: Add appropriate `time.sleep()` or use `pytest-timeout`
3. **File system tests**: Always use `tmp_path` fixture
4. **Mock issues**: Verify mock patches are correctly applied

### Debugging Commands

```bash
# Run single test with detailed output
pytest tests/unit/test_decorator.py::test_decorator_validation -v -s

# Drop into debugger on failure
pytest --pdb tests/unit/test_decorator.py

# Show print statements
pytest -s tests/unit/test_decorator.py
```

---

**Goal**: Maintain high confidence in Vald8's reliability across all supported LLM evaluation use cases.