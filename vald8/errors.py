"""
Error handling system for Vald8.

Provides structured error types with rich context information for better debugging
and user experience. Follows Pydantic's error handling patterns.
"""

from typing import Any, Dict, List, Optional, Union
import traceback


class Vald8Error(Exception):
    """Base exception class for all Vald8 errors."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        self.message = message
        self.context = context or {}
        super().__init__(message)
    
    def __str__(self) -> str:
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} ({context_str})"
        return self.message


class ValidationError(Vald8Error):
    """Raised when validation fails for inputs, datasets, or configurations."""
    
    def __init__(
        self, 
        message: str, 
        errors: Optional[List[Dict[str, Any]]] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        self.errors = errors or []
        super().__init__(message, context)
    
    @classmethod
    def from_pydantic(cls, pydantic_error: Exception, context: Optional[Dict[str, Any]] = None):
        """Create ValidationError from Pydantic ValidationError."""
        if hasattr(pydantic_error, 'errors'):
            errors = [
                {
                    "loc": error.get("loc", ()),
                    "msg": error.get("msg", ""),
                    "type": error.get("type", ""),
                    "input": error.get("input")
                }
                for error in pydantic_error.errors()
            ]
            message = f"Validation failed: {len(errors)} error(s)"
            return cls(message, errors, context)
        else:
            return cls(str(pydantic_error), context=context)


class DatasetValidationError(ValidationError):
    """Raised when dataset format or content validation fails."""
    
    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        line_number: Optional[int] = None,
        line_content: Optional[str] = None,
        errors: Optional[List[Dict[str, Any]]] = None
    ):
        context = {}
        if file_path:
            context["file_path"] = file_path
        if line_number is not None:
            context["line_number"] = line_number
        if line_content:
            context["line_content"] = line_content[:100] + "..." if len(line_content) > 100 else line_content
        
        super().__init__(message, errors, context)


class ConfigurationError(Vald8Error):
    """Raised when configuration is invalid or missing."""
    
    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        config_value: Optional[Any] = None,
        valid_options: Optional[List[str]] = None
    ):
        context = {}
        if config_key:
            context["config_key"] = config_key
        if config_value is not None:
            context["config_value"] = str(config_value)
        if valid_options:
            context["valid_options"] = valid_options
        
        super().__init__(message, context)


class EvaluationError(Vald8Error):
    """Raised when evaluation fails during function execution or metric calculation."""
    
    def __init__(
        self,
        message: str,
        function_name: Optional[str] = None,
        test_id: Optional[str] = None,
        original_error: Optional[Exception] = None
    ):
        context = {}
        if function_name:
            context["function_name"] = function_name
        if test_id:
            context["test_id"] = test_id
        if original_error:
            context["original_error"] = str(original_error)
            context["original_error_type"] = type(original_error).__name__
        
        super().__init__(message, context)
        self.original_error = original_error


class JudgeProviderError(Vald8Error):
    """Raised when judge provider fails to evaluate responses."""
    
    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        status_code: Optional[int] = None,
        response_text: Optional[str] = None
    ):
        context = {}
        if provider:
            context["provider"] = provider
        if model:
            context["model"] = model
        if status_code:
            context["status_code"] = status_code
        if response_text:
            # Truncate long response texts
            context["response_text"] = (
                response_text[:200] + "..." if len(response_text) > 200 else response_text
            )
        
        super().__init__(message, context)


class MetricCalculationError(Vald8Error):
    """Raised when metric calculation fails."""
    
    def __init__(
        self,
        message: str,
        metric_name: Optional[str] = None,
        test_id: Optional[str] = None,
        expected: Optional[Any] = None,
        actual: Optional[Any] = None
    ):
        context = {}
        if metric_name:
            context["metric_name"] = metric_name
        if test_id:
            context["test_id"] = test_id
        if expected is not None:
            context["expected"] = str(expected)[:100]
        if actual is not None:
            context["actual"] = str(actual)[:100]
        
        super().__init__(message, context)


def handle_evaluation_error(func_name: str, test_id: str, error: Exception) -> EvaluationError:
    """
    Handle errors that occur during function evaluation.
    
    Wraps the original error with context and returns a structured EvaluationError.
    """
    if isinstance(error, Vald8Error):
        return error
    
    # Get traceback for debugging
    tb = traceback.format_exc()
    
    return EvaluationError(
        message=f"Function evaluation failed: {str(error)}",
        function_name=func_name,
        test_id=test_id,
        original_error=error
    )


def format_validation_errors(errors: List[Dict[str, Any]]) -> str:
    """Format validation errors for human-readable output."""
    if not errors:
        return "No validation errors"
    
    formatted = []
    for error in errors:
        loc = " -> ".join(str(x) for x in error.get("loc", []))
        msg = error.get("msg", "Unknown error")
        formatted.append(f"  {loc}: {msg}")
    
    return "\n".join(formatted)