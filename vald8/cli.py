"""
Command-line interface for Vald8.

Provides CLI commands for running evaluations, managing results, and validating datasets.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from .config import config_manager
from .dataset import validate_dataset_format
from .errors import Vald8Error
from .results import ResultsManager


def validate_dataset_cmd(args) -> int:
    """Validate dataset format command."""
    try:
        warnings = validate_dataset_format(args.dataset)
        
        print(f"✅ Dataset validation completed: {args.dataset}")
        
        if warnings:
            print(f"\n⚠️  {len(warnings)} warnings found:")
            for warning in warnings:
                print(f"  - {warning}")
        else:
            print("No issues found.")
        
        return 0
        
    except Vald8Error as e:
        print(f"❌ Dataset validation failed: {e}")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1


def list_runs_cmd(args) -> int:
    """List evaluation runs command."""
    try:
        results_manager = ResultsManager(args.results_dir)
        runs = results_manager.list_runs(limit=args.limit)
        
        if not runs:
            print("No evaluation runs found.")
            return 0
        
        print(f"Recent evaluation runs (showing {len(runs)}):\n")
        
        for run in runs:
            status = "✅ PASSED" if run['passed'] else "❌ FAILED"
            timestamp = run.get('timestamp', '')[:19]  # Remove microseconds
            success_rate = run.get('success_rate', 0) * 100
            
            print(f"{status} {run['function_name']} ({timestamp})")
            print(f"  Run ID: {run['run_id'][:8]}")
            print(f"  Dataset: {run['dataset_path']}")
            print(f"  Tests: {run.get('total_tests', 0)} (Success: {success_rate:.1f}%)")
            print(f"  Results: {run['run_dir']}")
            print()
        
        return 0
        
    except Vald8Error as e:
        print(f"❌ Failed to list runs: {e}")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1


def show_run_cmd(args) -> int:
    """Show detailed run information command."""
    try:
        results_manager = ResultsManager()
        result = results_manager.load_results(args.run_dir)
        
        print(f"# Vald8 Evaluation Run Details\n")
        print(f"**Function:** {result.function_name}")
        print(f"**Dataset:** {result.dataset_path}")
        print(f"**Status:** {'✅ PASSED' if result.passed else '❌ FAILED'}")
        print(f"**Run ID:** {result.run_id}")
        print(f"**Timestamp:** {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Summary
        s = result.summary
        print(f"\n## Summary")
        print(f"- Total Tests: {s.total_tests}")
        print(f"- Passed: {s.passed_tests} ({s.success_rate:.1%})")
        print(f"- Failed: {s.failed_tests}")
        print(f"- Errors: {s.error_tests}")
        print(f"- Total Time: {s.total_time:.2f}s")
        
        # Metrics
        if s.metrics:
            print(f"\n## Metrics")
            for metric_name, stats in s.metrics.items():
                print(f"- **{metric_name.title()}:** {stats.get('mean', 0):.2f} (min: {stats.get('min', 0):.2f}, max: {stats.get('max', 0):.2f})")
        
        # Failed tests (if any)
        failed_tests = [t for t in result.tests if not t.passed]
        if failed_tests:
            print(f"\n## Failed Tests ({len(failed_tests)})")
            for test in failed_tests[:5]:  # Show first 5
                print(f"- **{test.test_id}:** {test.error or 'Metric thresholds not met'}")
            
            if len(failed_tests) > 5:
                print(f"  ... and {len(failed_tests) - 5} more")
        
        return 0
        
    except Vald8Error as e:
        print(f"❌ Failed to load run: {e}")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1


def cleanup_runs_cmd(args) -> int:
    """Cleanup old runs command."""
    try:
        results_manager = ResultsManager(args.results_dir)
        deleted_count = results_manager.cleanup_old_runs(keep_count=args.keep)
        
        if deleted_count > 0:
            print(f"✅ Cleaned up {deleted_count} old evaluation runs")
        else:
            print("No old runs to clean up")
        
        return 0
        
    except Vald8Error as e:
        print(f"❌ Cleanup failed: {e}")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    
    parser = argparse.ArgumentParser(
        prog='vald8',
        description='pytest for LLMs - Evaluate your LLM functions with structured datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  vald8 validate tests.jsonl           # Validate dataset format
  vald8 runs list                      # List recent evaluation runs  
  vald8 runs show ./runs/2024-01-01_*  # Show run details
  vald8 runs cleanup --keep 10         # Keep only 10 most recent runs

For more information, visit: https://github.com/osuthorpe/vald8
        """
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 0.1.0'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Validate command
    validate_parser = subparsers.add_parser(
        'validate',
        help='Validate dataset format'
    )
    validate_parser.add_argument(
        'dataset',
        help='Path to JSONL dataset file'
    )
    validate_parser.set_defaults(func=validate_dataset_cmd)
    
    # Runs command group
    runs_parser = subparsers.add_parser(
        'runs',
        help='Manage evaluation runs'
    )
    runs_subparsers = runs_parser.add_subparsers(dest='runs_command', help='Run management commands')
    
    # List runs
    list_parser = runs_subparsers.add_parser(
        'list',
        help='List recent evaluation runs'
    )
    list_parser.add_argument(
        '--limit',
        type=int,
        default=20,
        help='Maximum number of runs to show (default: 20)'
    )
    list_parser.add_argument(
        '--results-dir',
        default='runs',
        help='Results directory (default: runs)'
    )
    list_parser.set_defaults(func=list_runs_cmd)
    
    # Show run details
    show_parser = runs_subparsers.add_parser(
        'show',
        help='Show detailed run information'
    )
    show_parser.add_argument(
        'run_dir',
        help='Path to run directory'
    )
    show_parser.set_defaults(func=show_run_cmd)
    
    # Cleanup runs
    cleanup_parser = runs_subparsers.add_parser(
        'cleanup',
        help='Clean up old evaluation runs'
    )
    cleanup_parser.add_argument(
        '--keep',
        type=int,
        default=50,
        help='Number of recent runs to keep (default: 50)'
    )
    cleanup_parser.add_argument(
        '--results-dir',
        default='runs',
        help='Results directory (default: runs)'
    )
    cleanup_parser.set_defaults(func=cleanup_runs_cmd)
    
    return parser


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Show help if no command provided
    if not hasattr(args, 'func'):
        parser.print_help()
        return 0
    
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\n❌ Interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())