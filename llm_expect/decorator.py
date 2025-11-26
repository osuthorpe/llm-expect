"""
Core @llm_expect decorator and Runner implementation.

Provides the main decorator interface and evaluation runner that coordinates
all components to evaluate LLM functions.
"""

import concurrent.futures
import inspect
import time
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

from .config import config_manager
from .dataset import load_dataset
from .errors import EvaluationError, handle_evaluation_error
from .judges import create_judge_provider
from .metrics import MetricEvaluator
from .models import DatasetExample, EvaluationResult, TestResult, LLMExpectConfig
from .results import ResultsManager, calculate_summary_stats, generate_run_id

F = TypeVar('F', bound=Callable[..., Any])


class EvaluationRunner:
    """Coordinates evaluation of a function against a dataset."""
    
    def __init__(self, func: Callable, config: LLMExpectConfig):
        self.func = func
        self.config = config
        self.function_name = func.__name__
        
        # Initialize components
        self.results_manager = ResultsManager(config.results_dir)
        
        # Initialize judge provider if needed
        judge_provider = None
        tests = config.tests or []  # Defensive guard: handle None
        if config.judge and any(test in ["instruction_adherence", "safety", "custom_judge"] for test in tests):
            judge_provider = create_judge_provider(config.judge)
        
        self.metric_evaluator = MetricEvaluator(judge_provider)
    
    def run_evaluation(self) -> EvaluationResult:
        """
        Run complete evaluation of the function against the dataset.
        
        Returns:
            Complete evaluation result
            
        Raises:
            EvaluationError: If evaluation fails
        """
        run_id = generate_run_id()
        start_time = time.time()
        
        try:
            # Load dataset
            examples = load_dataset(
                self.config.dataset,
                sample_size=self.config.sample_size,
                shuffle=self.config.shuffle
            )
            
            if not examples:
                raise EvaluationError(
                    "Dataset is empty",
                    function_name=self.function_name
                )
            
            # Run tests
            test_results = []
            
            if self.config.parallel and not self.config.fail_fast and len(examples) > 1:
                # Run in parallel using ThreadPoolExecutor
                # Default to 5 workers or number of examples, whichever is smaller
                max_workers = min(len(examples), 10)
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Submit all tasks
                    future_to_example = {
                        executor.submit(self._evaluate_single_test, example): example 
                        for example in examples
                    }
                    
                    # Collect results as they complete
                    for future in concurrent.futures.as_completed(future_to_example):
                        try:
                            test_result = future.result()
                            test_results.append(test_result)
                        except Exception as e:
                            # Should be handled inside _evaluate_single_test, but just in case
                            example = future_to_example[future]
                            test_results.append(TestResult(
                                test_id=example.id,
                                input=example.input,
                                expected=example.expected,
                                actual=None,
                                error=f"Parallel execution error: {str(e)}",
                                metrics=[],
                                passed=False,
                                execution_time=0.0
                            ))
                            
                # Sort results by ID to maintain deterministic order
                # This assumes IDs are comparable, or we can use the original index
                # Let's map back to original order
                example_ids = [e.id for e in examples]
                test_results.sort(key=lambda x: example_ids.index(x.test_id) if x.test_id in example_ids else 999)
                
            else:
                # Run sequentially
                for example in examples:
                    test_result = self._evaluate_single_test(example)
                    test_results.append(test_result)
                    
                    # Check fail_fast
                    if self.config.fail_fast and not test_result.passed:
                        break
            
            # Calculate summary statistics
            summary = calculate_summary_stats(test_results)
            
            # Determine overall pass/fail
            overall_passed = (
                summary.error_tests == 0 and 
                summary.failed_tests == 0 and 
                summary.success_rate >= min(self.config.thresholds.values(), default=0.8)
            )
            
            # Create result object
            result = EvaluationResult(
                function_name=self.function_name,
                dataset_path=self.config.dataset,
                timestamp=datetime.now(),
                config=self.config,
                tests=test_results,
                summary=summary,
                passed=overall_passed,
                run_id=run_id
            )
            
            # Save results if configured
            if self.config.save_results:
                run_dir = self.results_manager.save_results(result)
                result.run_dir = run_dir
            
            return result
            
        except Exception as e:
            raise handle_evaluation_error(self.function_name, "evaluation", e)
    
    def _evaluate_single_test(self, example: DatasetExample) -> TestResult:
        """
        Evaluate a single test case.
        
        Args:
            example: Test case to evaluate
            
        Returns:
            Individual test result
        """
        test_start_time = time.time()
        
        try:
            # Call the function with appropriate input handling
            actual_output = self._call_function(example.input)
            execution_time = time.time() - test_start_time
            
            # Evaluate metrics
            metrics = self.metric_evaluator.evaluate_metrics(
                actual=actual_output,
                expected=example.expected,
                test_id=example.id,
                metric_names=self.config.tests,
                thresholds=self.config.thresholds
            )
            
            # Determine if test passed (all metrics must pass)
            test_passed = all(metric.passed for metric in metrics)
            
            return TestResult(
                test_id=example.id,
                input=example.input,
                expected=example.expected,
                actual=actual_output,
                error=None,
                metrics=metrics,
                passed=test_passed,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - test_start_time
            error_msg = str(e)
            
            return TestResult(
                test_id=example.id,
                input=example.input,
                expected=example.expected,
                actual=None,
                error=error_msg,
                metrics=[],
                passed=False,
                execution_time=execution_time
            )
    
    def _call_function(self, input_data: Union[str, int, float, bool, Dict[str, Any]]) -> Any:
        """
        Call the function with flexible input handling.
        
        Args:
            input_data: Input to pass to function (primitive or dict)
            
        Returns:
            Function output
            
        Raises:
            EvaluationError: If function call fails
        """
        try:
            # Get function signature for analysis
            sig = inspect.signature(self.func)
            params = list(sig.parameters.keys())
            
            # Handle different input patterns
            if isinstance(input_data, dict):
                # For dict inputs, try kwargs first, then single dict parameter
                try:
                    # Option 1: Spread dict as kwargs
                    return self.func(**input_data)
                except TypeError:
                    # Option 2: Pass dict as single parameter
                    if len(params) == 1:
                        return self.func(input_data)
                    else:
                        raise
            else:
                # For primitive inputs, pass directly
                if len(params) == 1:
                    return self.func(input_data)
                else:
                    raise EvaluationError(
                        f"Function {self.function_name} expects {len(params)} parameters, "
                        f"but got primitive input that can only fill 1 parameter",
                        function_name=self.function_name
                    )
                    
        except Exception as e:
            raise EvaluationError(
                f"Function call failed: {str(e)}",
                function_name=self.function_name,
                original_error=e
            )


class LLMExpectFunction:
    """Wrapper class that adds evaluation capabilities to a function."""
    
    def __init__(self, func: Callable, config: LLMExpectConfig):
        self.func = func
        self.config = config
        self.runner = EvaluationRunner(func, config)
        
        # Copy function metadata
        self.__name__ = func.__name__
        self.__doc__ = func.__doc__
        self.__module__ = func.__module__
        self.__qualname__ = getattr(func, '__qualname__', func.__name__)
        self.__annotations__ = getattr(func, '__annotations__', {})
    
    def __get__(self, obj, objtype=None):
        """
        Support instance method binding.
        When accessed from an instance, return a bound version of the function.
        """
        if obj is None:
            return self
            
        # Create a bound method
        bound_func = self.func.__get__(obj, objtype)
        
        # Return a new LLMExpectFunction wrapping the bound method, sharing the config
        # We need to ensure the new wrapper shares the same config object
        wrapper = LLMExpectFunction(bound_func, self.config)
        return wrapper
    
    def __call__(self, *args, **kwargs):
        """Normal function call - evaluates the wrapped function normally."""
        return self.func(*args, **kwargs)
    
    def run_eval(self) -> Dict[str, Any]:
        """
        Run evaluation against the configured dataset.
        
        Returns:
            Dictionary with evaluation results and summary information
        """
        result = self.runner.run_evaluation()
        
        # Return simplified result for easy consumption
        return {
            "passed": result.passed,
            "summary": {
                "total_tests": result.summary.total_tests,
                "passed_tests": result.summary.passed_tests,
                "failed_tests": result.summary.failed_tests,
                "error_tests": result.summary.error_tests,
                "success_rate": result.summary.success_rate,
                "metrics": result.summary.metrics
            },
            "run_id": result.run_id,
            "run_dir": result.run_dir,
            "timestamp": result.timestamp.isoformat(),
            "function_name": result.function_name,
            "dataset_path": result.dataset_path
        }
    
    def get_config(self) -> LLMExpectConfig:
        """Get the current configuration."""
        return self.config
    
    def update_config(self, **kwargs) -> None:
        """Update configuration parameters."""
        # Create new config with updates
        config_dict = self.config.model_dump()
        config_dict.update(kwargs)
        
        self.config = config_manager.create_config(**config_dict)
        
        # Recreate runner with new config
        self.runner = EvaluationRunner(self.func, self.config)


def llm_expect(
    dataset: str,
    tests: Optional[List[str]] = None,
    thresholds: Optional[Dict[str, float]] = None,
    judge_provider: Optional[str] = None,
    judge_model: Optional[str] = None,
    sample_size: Optional[int] = None,
    shuffle: bool = False,
    cache: bool = True,
    cache_dir: Optional[str] = None,
    results_dir: Optional[str] = None,
    fail_fast: bool = False,
    timeout: int = 60,
    **kwargs
) -> Callable[[F], LLMExpectFunction]:
    """
    Decorator to add evaluation capabilities to LLM functions.
    
    Args:
        dataset: Path to JSONL dataset file
        tests: List of metrics to evaluate (default: ["accuracy"])
        thresholds: Threshold values for pass/fail (default: {"accuracy": 0.8})
        judge_provider: LLM judge provider ("openai", "anthropic", "bedrock")
        judge_model: Model name for judge evaluation
        sample_size: Number of examples to sample from dataset
        shuffle: Whether to shuffle examples before sampling
        cache: Whether to cache evaluation results
        cache_dir: Directory for cache files
        results_dir: Directory to save evaluation results
        fail_fast: Stop evaluation on first test failure
        timeout: Function execution timeout in seconds
        **kwargs: Additional configuration parameters
    
    Returns:
        Decorated function with evaluation capabilities
    
    Example:
        @llm_expect(dataset="tests.jsonl")
        def my_llm_function(prompt: str) -> dict:
            return call_llm(prompt)
        
        # Normal usage
        result = my_llm_function("Hello")
        
        # Run evaluation
        eval_results = my_llm_function.run_eval()
        print(f"Passed: {eval_results['passed']}")
    """
    
    def decorator(func: F) -> LLMExpectFunction:
        # Create configuration
        config = config_manager.create_config(
            dataset=dataset,
            tests=tests,
            thresholds=thresholds,
            judge_provider=judge_provider,
            judge_model=judge_model,
            sample_size=sample_size,
            shuffle=shuffle,
            cache=cache,
            cache_dir=cache_dir,
            results_dir=results_dir,
            fail_fast=fail_fast,
            timeout=timeout,
            **kwargs
        )
        
        return LLMExpectFunction(func, config)
    
    return decorator


# Convenience aliases for common patterns
def pytest_for_llms(dataset: str, **kwargs) -> Callable[[F], LLMExpectFunction]:
    """
    Alias for llm_expect decorator emphasizing the pytest-like usage.
    
    Args:
        dataset: Path to JSONL dataset file  
        **kwargs: Same as llm_expect decorator
    
    Returns:
        Decorated function with evaluation capabilities
    """
    return llm_expect(dataset=dataset, **kwargs)


def llm_test(dataset: str, **kwargs) -> Callable[[F], LLMExpectFunction]:
    """
    Another alias for llm_expect decorator for concise usage.
    
    Args:
        dataset: Path to JSONL dataset file
        **kwargs: Same as llm_expect decorator
    
    Returns:
        Decorated function with evaluation capabilities  
    """
    return llm_expect(dataset=dataset, **kwargs)