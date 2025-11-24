"""
Pydantic models for Vald8 configuration and data structures.

Provides type-safe models with validation for all core Vald8 data structures.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class DatasetExample(BaseModel):
    """A single test example from a dataset."""
    
    id: str = Field(..., description="Unique identifier for this test case")
    input: Union[str, int, float, bool, Dict[str, Any]] = Field(
        ..., 
        description="Input data for the function (primitive or dict)"
    )
    expected: Dict[str, Any] = Field(
        ..., 
        description="Expected validation criteria"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Optional metadata for this example"
    )
    
    @field_validator('id')
    @classmethod
    def validate_id(cls, v):
        if not v or not v.strip():
            raise ValueError("ID cannot be empty")
        return v.strip()
    
    @field_validator('expected')
    @classmethod
    def validate_expected(cls, v):
        if not isinstance(v, dict) or not v:
            raise ValueError("Expected must be a non-empty dictionary")
        return v
    
    model_config = ConfigDict(extra="forbid")


class MetricResult(BaseModel):
    """Result of a single metric calculation."""
    
    name: str = Field(..., description="Name of the metric")
    score: float = Field(..., ge=0.0, le=1.0, description="Metric score (0.0 to 1.0)")
    passed: bool = Field(..., description="Whether this metric passed the threshold")
    threshold: float = Field(..., ge=0.0, le=1.0, description="Threshold used")
    details: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Additional metric-specific details"
    )
    
    model_config = ConfigDict(extra="forbid")


class TestResult(BaseModel):
    """Result of evaluating a single test case."""
    
    test_id: str = Field(..., description="ID of the test case")
    input: Union[str, int, float, bool, Dict[str, Any]] = Field(
        ..., 
        description="Input that was passed to the function"
    )
    expected: Dict[str, Any] = Field(..., description="Expected validation criteria")
    actual: Optional[Any] = Field(
        default=None, 
        description="Actual function output"
    )
    error: Optional[str] = Field(
        default=None, 
        description="Error message if function execution failed"
    )
    metrics: List[MetricResult] = Field(
        default_factory=list, 
        description="Metric results for this test"
    )
    passed: bool = Field(..., description="Whether all metrics passed")
    execution_time: float = Field(..., ge=0.0, description="Execution time in seconds")
    
    @field_validator('metrics')
    @classmethod
    def validate_metrics(cls, v):
        # Ensure metric names are unique
        names = [metric.name for metric in v]
        if len(names) != len(set(names)):
            raise ValueError("Metric names must be unique within a test result")
        return v
    
    model_config = ConfigDict(extra="forbid")


class EvaluationSummary(BaseModel):
    """Summary statistics for an evaluation run."""
    
    total_tests: int = Field(..., ge=0, description="Total number of tests")
    passed_tests: int = Field(..., ge=0, description="Number of tests that passed")
    failed_tests: int = Field(..., ge=0, description="Number of tests that failed")
    error_tests: int = Field(..., ge=0, description="Number of tests with errors")
    success_rate: float = Field(..., ge=0.0, le=1.0, description="Overall success rate")
    
    # Metric aggregations
    metrics: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Aggregated metric statistics (mean, std, min, max)"
    )
    
    # Timing information
    total_time: float = Field(..., ge=0.0, description="Total execution time in seconds")
    avg_time_per_test: float = Field(..., ge=0.0, description="Average time per test")
    
    @model_validator(mode='after')
    def validate_test_totals(self):
        total = self.total_tests
        passed = self.passed_tests
        failed = self.failed_tests
        error = self.error_tests
        
        if passed > total or failed > total or error > total:
            raise ValueError("Individual test counts cannot exceed total tests")
            
        if passed + failed + error != total:
            raise ValueError("Sum of passed, failed, and error tests must equal total tests")
        
        return self
    
    model_config = ConfigDict(extra="forbid")


class EvaluationResult(BaseModel):
    """Complete result of an evaluation run."""
    
    # Basic info
    function_name: str = Field(..., description="Name of the evaluated function")
    dataset_path: str = Field(..., description="Path to the dataset file")
    timestamp: datetime = Field(default_factory=datetime.now, description="When evaluation ran")
    
    # Configuration used
    config: 'Vald8Config' = Field(..., description="Configuration used for evaluation")
    
    # Results
    tests: List[TestResult] = Field(..., description="Individual test results")
    summary: EvaluationSummary = Field(..., description="Summary statistics")
    passed: bool = Field(..., description="Whether overall evaluation passed")
    
    # Metadata
    run_id: str = Field(..., description="Unique identifier for this run")
    run_dir: Optional[str] = Field(
        default=None, 
        description="Directory where results are saved"
    )
    
    model_config = ConfigDict(extra="forbid")


class JudgeConfig(BaseModel):
    """Configuration for LLM judge providers."""
    
    provider: Literal["openai", "anthropic", "bedrock"] = Field(
        ..., 
        description="Judge provider to use"
    )
    model: str = Field(..., description="Model name/ID to use")
    api_key: Optional[str] = Field(
        default=None, 
        description="API key (if not from environment)"
    )
    base_url: Optional[str] = Field(
        default=None, 
        description="Custom API base URL"
    )
    timeout: int = Field(
        default=30, 
        ge=1, 
        le=300, 
        description="Request timeout in seconds"
    )
    max_retries: int = Field(
        default=3, 
        ge=0, 
        le=10, 
        description="Maximum number of retries"
    )
    temperature: float = Field(
        default=0.0, 
        ge=0.0, 
        le=2.0, 
        description="Temperature for judge responses"
    )
    
    model_config = ConfigDict(extra="forbid")


class Vald8Config(BaseModel):
    """Main configuration for Vald8 evaluation."""
    
    # Dataset configuration
    dataset: str = Field(..., description="Path to the dataset file")
    sample_size: Optional[int] = Field(
        default=None, 
        ge=1, 
        description="Number of examples to sample (None for all)"
    )
    shuffle: bool = Field(
        default=False, 
        description="Whether to shuffle examples before sampling"
    )
    
    # Metrics configuration
    tests: List[str] = Field(
        default_factory=list, 
        description="List of metrics to evaluate (empty for auto-selection)"
    )
    thresholds: Dict[str, float] = Field(
        default_factory=lambda: {"accuracy": 0.8},
        description="Threshold values for each metric"
    )
    
    # Judge configuration (for LLM-based metrics)
    judge: Optional[JudgeConfig] = Field(
        default=None,
        description="Judge configuration for LLM-based evaluation"
    )
    
    # Caching
    cache: bool = Field(
        default=True, 
        description="Whether to cache evaluation results"
    )
    cache_dir: str = Field(
        default=".vald8_cache", 
        description="Directory for cache files"
    )
    
    # Output configuration
    results_dir: str = Field(
        default="runs", 
        description="Directory to save evaluation results"
    )
    save_results: bool = Field(
        default=True, 
        description="Whether to save detailed results to disk"
    )
    
    # Execution configuration
    fail_fast: bool = Field(
        default=False, 
        description="Stop evaluation on first test failure"
    )
    parallel: bool = Field(
        default=False, 
        description="Run evaluations in parallel (future)"
    )
    timeout: int = Field(
        default=60, 
        ge=1, 
        le=3600, 
        description="Timeout per function call in seconds"
    )
    
    @field_validator('tests')
    @classmethod
    def validate_tests(cls, v):
        if not v:  # Changed from "if v is None"
            return v
        valid_tests = {
            "accuracy", "schema_fidelity", "instruction_adherence", 
            "safety", "semantic_similarity", "custom_judge"  # Added custom_judge
        }
        invalid = set(v) - valid_tests
        if invalid:
            raise ValueError(f"Invalid tests: {invalid}. Valid tests: {valid_tests}")
        return v
    
    @field_validator('dataset')
    @classmethod
    def validate_dataset(cls, v):
        if not v.endswith('.jsonl'):
            raise ValueError("Dataset file must have .jsonl extension")
        return v
    
    @model_validator(mode='after')
    def validate_thresholds(self):
        # Ensure thresholds exist for all tests that need them
        if self.tests:
            for test in self.tests:
                if test not in self.thresholds:
                    self.thresholds[test] = 0.8  # Default threshold
        
        # Validate threshold values
        for test, threshold in self.thresholds.items():
            if not 0.0 <= threshold <= 1.0:
                raise ValueError(f"Threshold for {test} must be between 0.0 and 1.0")
        
        return self
    
    model_config = ConfigDict(extra="forbid")


# Update forward references
EvaluationResult.model_rebuild()