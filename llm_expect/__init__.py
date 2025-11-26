"""
LLM Expect: Lightweight Python SDK for automated LLM evaluation.

pytest for LLMs - A developer-first evaluation framework for testing LLM functions 
with structured datasets, configurable metrics, and threshold-based validation.
"""

# Core decorator and convenience functions
from .decorator import llm_expect, pytest_for_llms, llm_test

# Data models
from .models import (
    DatasetExample, 
    EvaluationResult, 
    EvaluationSummary,
    TestResult,
    MetricResult, 
    LLMExpectConfig,
    JudgeConfig
)

# Error types
from .errors import (
    LLMExpectError,
    ValidationError, 
    DatasetValidationError, 
    ConfigurationError,
    EvaluationError,
    JudgeProviderError,
    MetricCalculationError
)

# Utility functions
from .dataset import load_dataset, validate_dataset_format
from .config import config_manager
from .results import ResultsManager

__version__ = "0.1.9"
__author__ = "Alex Thorpe"
__license__ = "MIT"
__description__ = "LLM Expect is a minimalist, developer-first SDK for testing LLM-powered Python functions using structured JSONL datasets."

__all__ = [
    # Core decorator
    "llm_expect",
    "pytest_for_llms", 
    "llm_test",
    
    # Data models
    "DatasetExample", 
    "EvaluationResult",
    "EvaluationSummary",
    "TestResult",
    "MetricResult",
    'LLMExpectConfig',
    "JudgeConfig",
    
    # Error types
    "LLMExpectError",
    "ValidationError",
    "DatasetValidationError", 
    "ConfigurationError",
    "EvaluationError",
    "JudgeProviderError",
    "MetricCalculationError",
    
    # Utilities
    "load_dataset",
    "validate_dataset_format",
    "config_manager",
    "ResultsManager",
]