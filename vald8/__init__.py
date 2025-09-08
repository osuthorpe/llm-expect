"""
Vald8: Lightweight Python SDK for automated LLM evaluation.

A developer-first evaluation framework for testing LLM functions with structured
datasets, configurable metrics, and threshold-based validation.
"""

from .decorator import vald8
from .models import DatasetExample, EvaluationResult, MetricResult, Vald8Config
from .errors import ValidationError, DatasetValidationError, ConfigurationError

__version__ = "0.1.0"
__all__ = [
    "vald8",
    "DatasetExample", 
    "EvaluationResult",
    "MetricResult",
    "Vald8Config",
    "ValidationError",
    "DatasetValidationError", 
    "ConfigurationError",
]