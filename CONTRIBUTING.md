# Contributing to Vald8

Thank you for your interest in contributing to Vald8! This guide will help you get started.

## 🚀 Quick Start

### Development Setup

1. **Clone and setup**:
```bash
git clone https://github.com/osuthorpe/vald8
cd vald8
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install in development mode**:
```bash
pip install -e ".[dev]"
```

3. **Run tests to verify setup**:
```bash
pytest tests/ -v
```

### Development Commands

```bash
# Format code
black vald8/ tests/

# Lint code
ruff check vald8/ tests/

# Type checking
mypy vald8/

# Run all quality checks
black vald8/ tests/ && ruff check vald8/ tests/ && mypy vald8/ && pytest
```

## 📋 Contribution Process

### 1. Find or Create an Issue

- Check existing [issues](https://github.com/osuthorpe/vald8/issues)
- For bugs: include reproduction steps and error messages
- For features: explain the use case and expected behavior

### 2. Fork and Branch

```bash
# Fork on GitHub, then clone your fork
git clone https://github.com/your-username/vald8
cd vald8

# Create feature branch
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

### 3. Development Workflow

**Follow the Test-Driven Development process:**

1. **Write failing tests first**:
```python
def test_new_feature():
    """Test should initially fail."""
    result = new_feature("input")
    assert result == "expected_output"
```

2. **Run tests to confirm failure**:
```bash
pytest tests/test_new_feature.py -v
```

3. **Implement minimal code to pass**:
```python
def new_feature(input_data):
    return "expected_output"  # Minimal implementation
```

4. **Refactor and improve**:
   - Add error handling
   - Improve performance
   - Add comprehensive tests

### 4. Code Quality Standards

**File Organization:**
- Keep files under 300 lines
- One class per file for complex components
- Clear, descriptive module names

**Code Style:**
- Follow [Black](https://black.readthedocs.io/) formatting
- Use type hints on all public functions
- Write docstrings for all public APIs

**Example:**
```python
from typing import Dict, List, Optional

def evaluate_accuracy(
    output: str, 
    expected: Dict[str, Any], 
    threshold: float = 0.8
) -> Dict[str, Any]:
    """
    Evaluate accuracy of model output against expected results.
    
    Args:
        output: The model's response
        expected: Dictionary with expected values/patterns  
        threshold: Minimum score to pass (0.0-1.0)
        
    Returns:
        Dictionary with score, passed status, and details
        
    Example:
        >>> evaluate_accuracy("Paris", {"reference": "Paris"}, 0.8)
        {"score": 1.0, "passed": True, "details": "exact_match"}
    """
    # Implementation here
    pass
```

**Testing Requirements:**
- Write tests for all new functions
- Include edge cases and error conditions
- Use descriptive test names: `test_decorator_fails_on_missing_dataset()`
- Target 90%+ code coverage

### 5. Commit and Push

**Commit Message Format:**
```
type(scope): brief description

More detailed explanation if needed.

Fixes #123
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix  
- `docs`: Documentation changes
- `test`: Adding tests
- `refactor`: Code restructuring
- `perf`: Performance improvements

**Examples:**
```bash
git commit -m "feat(metrics): add semantic similarity metric

Implement semantic similarity using sentence transformers.
Supports threshold-based evaluation with configurable models.

Fixes #45"

git commit -m "fix(dataset): handle malformed JSON gracefully

DatasetLoader now continues processing after JSON decode errors,
collecting all errors for batch reporting.

Fixes #67"
```

### 6. Pull Request

**Before submitting:**
```bash
# Run full test suite
pytest -v

# Ensure code quality
black vald8/ tests/
ruff check vald8/ tests/  
mypy vald8/

# Update documentation if needed
# Add entry to CHANGELOG.md if significant change
```

**PR Description Template:**
```markdown
## Summary
Brief description of changes

## Changes Made
- [ ] Added new feature X
- [ ] Fixed bug Y  
- [ ] Updated documentation Z

## Testing
- [ ] Added unit tests
- [ ] Added integration tests
- [ ] Manual testing completed

## Breaking Changes
None / List any breaking changes

Fixes #issue-number
```

## 🎯 Contribution Areas

### High Priority
- **Core metrics**: New evaluation metrics (semantic similarity, factual accuracy)
- **Judge providers**: Additional LLM providers (Cohere, Gemini, local models)
- **Performance**: Optimization for large datasets
- **Documentation**: More examples and tutorials

### Medium Priority  
- **CLI enhancements**: Better progress reporting, interactive mode
- **Caching**: Advanced caching strategies
- **Reporting**: Additional output formats (PDF, dashboard)
- **Integration**: MLflow, Weights & Biases connectors

### Low Priority
- **Web UI**: Browser-based evaluation interface
- **Async support**: Asynchronous evaluation execution
- **Distributed**: Multi-machine evaluation

## 🐛 Bug Reports

**Include in bug reports:**

1. **Environment**:
   - Python version
   - Vald8 version
   - Operating system

2. **Reproduction**:
   - Minimal code example
   - Dataset sample (if relevant)
   - Full error traceback

3. **Expected vs Actual**:
   - What should happen
   - What actually happens

**Example:**
```markdown
**Environment:**
- Python 3.9.0
- Vald8 0.1.0  
- macOS 12.0

**Code:**
```python
@vald8(dataset="test.jsonl", tests=["accuracy"])
def my_func(input_data):
    return "response"
```

**Error:**
```
ValidationError: Dataset file not found: test.jsonl
```

**Expected:** Clear error message with suggestions
**Actual:** Generic file not found error
```

## 💡 Feature Requests

**Include in feature requests:**

1. **Use case**: Why is this needed?
2. **Proposed API**: How should it work?
3. **Examples**: Show usage scenarios
4. **Alternatives**: Other approaches considered

## 📝 Documentation

**Types of documentation:**
- **Code comments**: For complex logic
- **Docstrings**: For all public functions
- **README updates**: For new features
- **Examples**: In `examples/` directory
- **Blog posts**: Advanced use cases

## 🎉 Recognition

Contributors will be:
- Listed in `CONTRIBUTORS.md`
- Mentioned in release notes
- Acknowledged in relevant documentation

## ❓ Questions?

- **General questions**: Open a [Discussion](https://github.com/osuthorpe/vald8/discussions)
- **Bug reports**: Create an [Issue](https://github.com/osuthorpe/vald8/issues)
- **Security concerns**: See [SECURITY.md](SECURITY.md)

---

**Happy contributing! 🎉**