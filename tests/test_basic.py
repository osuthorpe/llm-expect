"""
Basic functionality tests for Vald8.

Tests core components to ensure basic functionality works correctly.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from llm_expect import llm_expect
from llm_expect.models import EvaluationResult, TestResult

from llm_expect import (
    DatasetExample,
    ValidationError,
    DatasetValidationError,
    ConfigurationError,
    EvaluationError,
    validate_dataset_format
)
from llm_expect.dataset import load_dataset
from pydantic import ValidationError as PydanticValidationError


def test_dataset_example_creation():
    """Test DatasetExample model creation and validation."""
    
    # Valid example
    example = DatasetExample(
        id="test1",
        input="What is 2+2?",
        expected={"reference": "4"}
    )
    
    assert example.id == "test1"
    assert example.input == "What is 2+2?"
    assert example.expected == {"reference": "4"}
    
    # With metadata
    example_with_meta = DatasetExample(
        id="test2",
        input={"question": "What is the capital of France?"},
        expected={"contains": ["Paris"]},
        metadata={"category": "geography"}
    )
    
    assert example_with_meta.metadata == {"category": "geography"}


def test_dataset_example_validation():
    """Test DatasetExample validation rules."""
    
    # Empty ID should fail
    with pytest.raises(PydanticValidationError):
        DatasetExample(id="", input="test", expected={"reference": "test"})
    
    # Empty expected should fail
    with pytest.raises(PydanticValidationError):
        DatasetExample(id="test", input="test", expected={})


def test_basic_decorator():
    """Test basic @vald8 decorator functionality."""
    
    # Create temporary dataset file
    test_dataset = Path("test_basic.jsonl")
    with open(test_dataset, 'w') as f:
        f.write('{"id": "test1", "input": "2+2", "expected": {"reference": "4"}}\n')
        f.write('{"id": "test2", "input": "3+3", "expected": {"reference": "6"}}\n')
    
    try:
        # Simple function with decorator
        @llm_expect(dataset=str(test_dataset))
        def simple_math(expr: str) -> str:
            """Simple math function for testing."""
            if expr == "2+2":
                return "4"
            elif expr == "3+3":
                return "6"
            else:
                return "unknown"
        
        # Test normal function usage
        assert simple_math("2+2") == "4"
        assert simple_math("3+3") == "6"
        
        # Test that function has evaluation capabilities
        assert hasattr(simple_math, 'run_eval')
        assert hasattr(simple_math, 'get_config')
        
        # Test config access
        config = simple_math.get_config()
        assert config.dataset == str(test_dataset)
        assert config.tests == []  # Changed from None to []
        
    finally:
        # Cleanup
        if test_dataset.exists():
            test_dataset.unlink()


def test_function_call_patterns():
    """Test different function calling patterns."""
    
    # Create dataset with different input types
    test_dataset = Path("test_patterns.jsonl")
    with open(test_dataset, 'w') as f:
        f.write('{"id": "primitive", "input": "hello", "expected": {"contains": ["hello"]}}\n')
        f.write('{"id": "dict", "input": {"name": "Alice", "age": 25}, "expected": {"contains": ["Alice", "25"]}}\n')
    
    try:
        # Function that handles primitive input
        @llm_expect(dataset=str(test_dataset), sample_size=1)
        def echo_function(text: str) -> str:
            return f"Echo: {text}"
        
        # Test normal call
        result = echo_function("test")
        assert "test" in result
        
        # Function that handles dict input  
        @llm_expect(dataset=str(test_dataset), sample_size=1)
        def dict_function(name: str, age: int) -> str:
            return f"{name} is {age} years old"
        
        # This should work with dict spreading
        assert hasattr(dict_function, 'run_eval')
        
    finally:
        if test_dataset.exists():
            test_dataset.unlink()


def test_dataset_loading():
    """Test dataset loading functionality."""
    
    # Create test dataset
    test_dataset = Path("test_loading.jsonl")
    with open(test_dataset, 'w') as f:
        f.write('{"id": "test1", "input": "hello", "expected": {"reference": "world"}}\n')
        f.write('{"id": "test2", "input": "foo", "expected": {"reference": "bar"}}\n')
        f.write('{"id": "test3", "input": "ping", "expected": {"reference": "pong"}}\n')
    
    try:
        # Load all examples
        examples = load_dataset(str(test_dataset))
        assert len(examples) == 3
        assert all(isinstance(ex, DatasetExample) for ex in examples)
        
        # Load with sampling
        examples_sampled = load_dataset(str(test_dataset), sample_size=2)
        assert len(examples_sampled) == 2
        
        # Load with shuffle
        examples_shuffled = load_dataset(str(test_dataset), shuffle=True)
        assert len(examples_shuffled) == 3
        
    finally:
        if test_dataset.exists():
            test_dataset.unlink()


def test_dataset_validation():
    """Test dataset format validation."""
    
    # Valid dataset
    valid_dataset = Path("test_valid.jsonl")
    with open(valid_dataset, 'w') as f:
        f.write('{"id": "test1", "input": "hello", "expected": {"reference": "world"}}\n')
        f.write('{"id": "test2", "input": "foo", "expected": {"contains": ["bar"]}}\n')
    
    try:
        warnings = validate_dataset_format(str(valid_dataset))
        assert isinstance(warnings, list)  # Should return list of warnings
        
    finally:
        if valid_dataset.exists():
            valid_dataset.unlink()
    
    # Invalid dataset - duplicate IDs
    invalid_dataset = Path("test_invalid.jsonl")
    with open(invalid_dataset, 'w') as f:
        f.write('{"id": "test1", "input": "hello", "expected": {"reference": "world"}}\n')
        f.write('{"id": "test1", "input": "duplicate", "expected": {"reference": "error"}}\n')
    
    try:
        with pytest.raises(DatasetValidationError) as exc_info:
            validate_dataset_format(str(invalid_dataset))
        
        assert "Duplicate test ID" in str(exc_info.value)
        
    finally:
        if invalid_dataset.exists():
            invalid_dataset.unlink()
    
    # Invalid JSON
    bad_json_dataset = Path("test_bad_json.jsonl")
    with open(bad_json_dataset, 'w') as f:
        f.write('{"id": "test1", "input": "hello", "expected": {"reference": "world"}}\n')
        f.write('{"id": "test2", "input": "foo", "expected": {invalid json}\n')
    
    try:
        with pytest.raises(DatasetValidationError) as exc_info:
            validate_dataset_format(str(bad_json_dataset))
        
        assert "Invalid JSON" in str(exc_info.value)
        
    finally:
        if bad_json_dataset.exists():
            bad_json_dataset.unlink()


def test_nonexistent_dataset():
    """Test handling of nonexistent dataset files."""
    
    with pytest.raises(DatasetValidationError) as exc_info:
        load_dataset("nonexistent.jsonl")
    
    assert "not found" in str(exc_info.value).lower()


def test_decorator_aliases():
    """Test decorator aliases work correctly."""
    
    from llm_expect import pytest_for_llms, llm_test
    
    test_dataset = Path("test_aliases.jsonl") 
    with open(test_dataset, 'w') as f:
        f.write('{"id": "test1", "input": "hello", "expected": {"reference": "world"}}\n')
    
    try:
        # Test pytest_for_llms alias
        @pytest_for_llms(dataset=str(test_dataset))
        def test_func1(x):
            return "world"
        
        assert hasattr(test_func1, 'run_eval')
        
        # Test llm_test alias  
        @llm_test(dataset=str(test_dataset))
        def test_func2(x):
            return "world"
        
        assert hasattr(test_func2, 'run_eval')
        
    finally:
        if test_dataset.exists():
            test_dataset.unlink()


if __name__ == "__main__":
    pytest.main([__file__])