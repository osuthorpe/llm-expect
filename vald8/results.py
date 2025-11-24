"""
Result serialization and reporting for Vald8.

Handles saving evaluation results to disk, generating reports, and managing result data.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .errors import Vald8Error
from .models import EvaluationResult, EvaluationSummary, MetricResult, TestResult


class ResultsManager:
    """Manages evaluation results storage and reporting."""
    
    def __init__(self, results_dir: str = "runs"):
        """Initialize results manager."""
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def save_results(self, result: EvaluationResult) -> str:
        """
        Save evaluation results to disk with structured file organization.
        
        Args:
            result: Complete evaluation result to save
            
        Returns:
            Path to the saved results directory
            
        Raises:
            Vald8Error: If saving fails
        """
        try:
            # Get session ID for grouping
            from .results import get_session_id
            session_id = get_session_id()
            
            # Create master session folder: {date}_{session_id}
            session_date = result.timestamp.strftime("%Y-%m-%d")
            session_dir = self.results_dir / f"{session_date}_{session_id[:8]}"
            session_dir.mkdir(parents=True, exist_ok=True)
            
            # Create function-specific folder inside session
            run_dir = session_dir / result.function_name
            run_dir.mkdir(parents=True, exist_ok=True)
            
            # Save detailed results as JSONL (one test per line)
            results_file = run_dir / "results.jsonl"
            with open(results_file, 'w', encoding='utf-8') as f:
                for test in result.tests:
                    # Convert to dict and write as JSON line
                    test_dict = test.model_dump()
                    f.write(json.dumps(test_dict) + '\n')
            
            # Save summary as JSON
            summary_file = run_dir / "summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                summary_dict = result.summary.model_dump()
                json.dump(summary_dict, f, indent=2, default=str)
            
            # Save metadata (config and run info)
            metadata_file = run_dir / "metadata.json"
            metadata = {
                "function_name": result.function_name,
                "dataset_path": result.dataset_path,
                "timestamp": result.timestamp.isoformat(),
                "run_id": result.run_id,
                "config": result.config.model_dump(),
                "passed": result.passed,
                "total_tests": len(result.tests)
            }
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            # Generate human-readable report
            report_file = run_dir / "report.txt"
            self._generate_text_report(result, report_file)
            
            # Update the result object with the run directory
            result.run_dir = str(run_dir)
            
            return str(run_dir)
            
        except Exception as e:
            raise Vald8Error(f"Failed to save results: {str(e)}")
    
    def _generate_text_report(self, result: EvaluationResult, report_file: Path) -> None:
        """Generate a human-readable text report."""
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"# Vald8 Evaluation Report\n\n")
            f.write(f"**Function:** {result.function_name}\n")
            f.write(f"**Dataset:** {result.dataset_path}\n")
            f.write(f"**Timestamp:** {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Run ID:** {result.run_id}\n")
            f.write(f"**Overall Result:** {'✅ PASSED' if result.passed else '❌ FAILED'}\n\n")
            
            # Summary statistics
            f.write("## Summary\n\n")
            summary = result.summary
            f.write(f"- **Total Tests:** {summary.total_tests}\n")
            f.write(f"- **Passed:** {summary.passed_tests} ({summary.success_rate:.1%})\n")
            f.write(f"- **Failed:** {summary.failed_tests}\n")
            f.write(f"- **Errors:** {summary.error_tests}\n")
            f.write(f"- **Total Time:** {summary.total_time:.2f}s\n")
            f.write(f"- **Avg Time/Test:** {summary.avg_time_per_test:.2f}s\n\n")
            
            # Metric breakdown
            if summary.metrics:
                f.write("## Metrics\n\n")
                for metric_name, stats in summary.metrics.items():
                    f.write(f"### {metric_name.title()}\n")
                    f.write(f"- **Mean Score:** {stats.get('mean', 0):.2f}\n")
                    f.write(f"- **Min Score:** {stats.get('min', 0):.2f}\n")
                    f.write(f"- **Max Score:** {stats.get('max', 0):.2f}\n")
                    if 'std' in stats:
                        f.write(f"- **Std Dev:** {stats['std']:.2f}\n")
                    f.write("\n")
            
            # Configuration
            f.write("## Configuration\n\n")
            config = result.config
            tests_str = ', '.join(config.tests) if config.tests else "Auto-selected"
            f.write(f"- **Tests:** {tests_str}\n")
            f.write(f"- **Thresholds:** {config.thresholds}\n")
            f.write(f"- **Sample Size:** {config.sample_size or 'All'}\n")
            f.write(f"- **Cache:** {config.cache}\n")
            f.write(f"- **Timeout:** {config.timeout}s\n")
            if config.judge:
                f.write(f"- **Judge:** {config.judge.provider} ({config.judge.model})\n")
            f.write("\n")
            
            # Failed tests details
            failed_tests = [test for test in result.tests if not test.passed]
            if failed_tests:
                f.write("## Failed Tests\n\n")
                for test in failed_tests[:10]:  # Limit to first 10
                    f.write(f"### {test.test_id}\n")
                    if test.error:
                        f.write(f"**Error:** {test.error}\n")
                    else:
                        failed_metrics = [m for m in test.metrics if not m.passed]
                        for metric in failed_metrics:
                            f.write(f"**{metric.name}:** {metric.score:.2f} (threshold: {metric.threshold:.2f})\n")
                    f.write("\n")
                
                if len(failed_tests) > 10:
                    f.write(f"... and {len(failed_tests) - 10} more failed tests.\n\n")
    
    def load_results(self, run_dir: str) -> EvaluationResult:
        """
        Load evaluation results from disk.
        
        Args:
            run_dir: Path to the results directory
            
        Returns:
            Reconstructed EvaluationResult object
            
        Raises:
            Vald8Error: If loading fails
        """
        try:
            run_path = Path(run_dir)
            
            if not run_path.exists():
                raise Vald8Error(f"Results directory not found: {run_dir}")
            
            # Load metadata
            metadata_file = run_path / "metadata.json"
            if not metadata_file.exists():
                raise Vald8Error(f"Metadata file not found: {metadata_file}")
            
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Load summary
            summary_file = run_path / "summary.json"
            if not summary_file.exists():
                raise Vald8Error(f"Summary file not found: {summary_file}")
            
            with open(summary_file, 'r') as f:
                summary_data = json.load(f)
            
            summary = EvaluationSummary(**summary_data)
            
            # Load test results
            results_file = run_path / "results.jsonl"
            if not results_file.exists():
                raise Vald8Error(f"Results file not found: {results_file}")
            
            tests = []
            with open(results_file, 'r') as f:
                for line in f:
                    if line.strip():
                        test_data = json.loads(line)
                        
                        # Reconstruct MetricResult objects
                        metrics = []
                        for metric_data in test_data.get('metrics', []):
                            metrics.append(MetricResult(**metric_data))
                        test_data['metrics'] = metrics
                        
                        tests.append(TestResult(**test_data))
            
            # Reconstruct config (simplified - some validation may be lost)
            from .models import Vald8Config, JudgeConfig
            config_data = metadata['config']
            if 'judge' in config_data and config_data['judge']:
                config_data['judge'] = JudgeConfig(**config_data['judge'])
            config = Vald8Config(**config_data)
            
            # Create result object
            result = EvaluationResult(
                function_name=metadata['function_name'],
                dataset_path=metadata['dataset_path'],
                timestamp=datetime.fromisoformat(metadata['timestamp']),
                config=config,
                tests=tests,
                summary=summary,
                passed=metadata['passed'],
                run_id=metadata['run_id'],
                run_dir=str(run_path)
            )
            
            return result
            
        except Exception as e:
            raise Vald8Error(f"Failed to load results from {run_dir}: {str(e)}")
    
    def list_runs(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        List recent evaluation runs.
        
        Args:
            limit: Maximum number of runs to return
            
        Returns:
            List of run summaries with basic info
        """
        runs = []
        
        try:
            # Get all run directories, sorted by modification time (newest first)
            run_dirs = [d for d in self.results_dir.iterdir() if d.is_dir()]
            run_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            for run_dir in run_dirs[:limit]:
                try:
                    metadata_file = run_dir / "metadata.json"
                    if metadata_file.exists():
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        
                        # Load summary for additional info
                        summary_file = run_dir / "summary.json"
                        summary_info = {}
                        if summary_file.exists():
                            with open(summary_file, 'r') as f:
                                summary_data = json.load(f)
                                summary_info = {
                                    'total_tests': summary_data.get('total_tests', 0),
                                    'success_rate': summary_data.get('success_rate', 0),
                                    'total_time': summary_data.get('total_time', 0)
                                }
                        
                        runs.append({
                            'run_id': metadata.get('run_id', ''),
                            'function_name': metadata.get('function_name', ''),
                            'dataset_path': metadata.get('dataset_path', ''),
                            'timestamp': metadata.get('timestamp', ''),
                            'passed': metadata.get('passed', False),
                            'run_dir': str(run_dir),
                            **summary_info
                        })
                        
                except Exception:
                    # Skip runs with corrupted metadata
                    continue
            
            return runs
            
        except Exception as e:
            raise Vald8Error(f"Failed to list runs: {str(e)}")
    
    def cleanup_old_runs(self, keep_count: int = 50) -> int:
        """
        Clean up old evaluation runs, keeping only the most recent ones.
        
        Args:
            keep_count: Number of recent runs to keep
            
        Returns:
            Number of runs deleted
        """
        try:
            run_dirs = [d for d in self.results_dir.iterdir() if d.is_dir()]
            run_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            deleted_count = 0
            
            # Delete runs beyond the keep_count
            for run_dir in run_dirs[keep_count:]:
                try:
                    import shutil
                    shutil.rmtree(run_dir)
                    deleted_count += 1
                except Exception:
                    # Continue if we can't delete a specific run
                    continue
            
            return deleted_count
            
        except Exception as e:
            raise Vald8Error(f"Failed to cleanup old runs: {str(e)}")


def calculate_summary_stats(tests: List[TestResult]) -> EvaluationSummary:
    """
    Calculate summary statistics from test results.
    
    Args:
        tests: List of individual test results
        
    Returns:
        EvaluationSummary with aggregated statistics
    """
    if not tests:
        return EvaluationSummary(
            total_tests=0,
            passed_tests=0,
            failed_tests=0,
            error_tests=0,
            success_rate=0.0,
            total_time=0.0,
            avg_time_per_test=0.0
        )
    
    # Basic counts
    total_tests = len(tests)
    passed_tests = sum(1 for test in tests if test.passed and not test.error)
    error_tests = sum(1 for test in tests if test.error)
    failed_tests = total_tests - passed_tests - error_tests
    
    success_rate = passed_tests / total_tests if total_tests > 0 else 0.0
    
    # Timing statistics
    total_time = sum(test.execution_time for test in tests)
    avg_time_per_test = total_time / total_tests if total_tests > 0 else 0.0
    
    # Metric aggregations
    metrics_stats = {}
    
    # Get all unique metric names
    all_metric_names = set()
    for test in tests:
        if not test.error:  # Only include tests without errors
            all_metric_names.update(metric.name for metric in test.metrics)
    
    # Calculate stats for each metric
    for metric_name in all_metric_names:
        scores = []
        for test in tests:
            if not test.error:
                for metric in test.metrics:
                    if metric.name == metric_name:
                        scores.append(metric.score)
                        break
        
        if scores:
            import statistics
            metrics_stats[metric_name] = {
                "mean": statistics.mean(scores),
                "min": min(scores),
                "max": max(scores),
                "std": statistics.stdev(scores) if len(scores) > 1 else 0.0
            }
    
    return EvaluationSummary(
        total_tests=total_tests,
        passed_tests=passed_tests,
        failed_tests=failed_tests,
        error_tests=error_tests,
        success_rate=success_rate,
        metrics=metrics_stats,
        total_time=total_time,
        avg_time_per_test=avg_time_per_test
    )


def generate_run_id() -> str:
    """Generate a unique run ID."""
    return str(uuid.uuid4())


# Global session ID for grouping runs from the same script execution
_current_session_id: Optional[str] = None


def get_session_id() -> str:
    """
    Get or create the current session ID.
    
    All functions evaluated in the same script run will share this session ID,
    allowing their results to be grouped together in a single master folder.
    """
    global _current_session_id
    if _current_session_id is None:
        _current_session_id = str(uuid.uuid4())
    return _current_session_id


def reset_session_id() -> None:
    """Reset the session ID (useful for testing or manual control)."""
    global _current_session_id
    _current_session_id = None