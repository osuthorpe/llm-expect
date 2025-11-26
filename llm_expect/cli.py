"""
Command-line interface for llm-expect.

Provides CLI commands for running evaluations, managing results, and validating datasets.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from .config import config_manager
from .dataset import validate_dataset_format
from .errors import LLMExpectError
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
        
    except LLMExpectError as e:
        print(f"❌ Dataset validation failed: {e}")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1


def list_runs_cmd(args) -> int:
    """List evaluation runs command."""
    console = Console()
    try:
        results_manager = ResultsManager(args.results_dir)
        runs = results_manager.list_runs(limit=args.limit)
        
        if not runs:
            console.print("[yellow]No evaluation runs found.[/yellow]")
            return 0
        
        console.print(f"\n[bold]Recent Evaluation Runs[/bold] (showing {len(runs)})\n")
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Status", justify="center")
        table.add_column("Function", style="cyan")
        table.add_column("Date", style="dim")
        table.add_column("Success Rate", justify="right")
        table.add_column("Tests", justify="right")
        table.add_column("Run ID", style="dim")
        
        for run in runs:
            status = "[green]PASSED[/green]" if run['passed'] else "[red]FAILED[/red]"
            timestamp = run.get('timestamp', '')[:19]
            success_rate = run.get('success_rate', 0) * 100
            
            # Colorize success rate
            rate_str = f"{success_rate:.1f}%"
            if success_rate >= 90:
                rate_str = f"[green]{rate_str}[/green]"
            elif success_rate >= 70:
                rate_str = f"[yellow]{rate_str}[/yellow]"
            else:
                rate_str = f"[red]{rate_str}[/red]"
            
            table.add_row(
                status,
                run['function_name'],
                timestamp,
                rate_str,
                str(run.get('total_tests', 0)),
                run['run_id'][:8]
            )
            
        console.print(table)
        console.print()
        return 0
        
    except LLMExpectError as e:
        console.print(f"[bold red]Failed to list runs:[/bold red] {e}")
        return 1
    except Exception as e:
        console.print(f"[bold red]Unexpected error:[/bold red] {e}")
        return 1


def show_run_cmd(args) -> int:
    """Show detailed run information command."""
    console = Console()
    try:
        results_manager = ResultsManager()
        result = results_manager.load_results(args.run_dir)
        
        # Header
        status_color = "green" if result.passed else "red"
        status_text = "PASSED" if result.passed else "FAILED"
        
        console.print(Panel(
            f"[bold {status_color}]{status_text}[/bold {status_color}] - {result.function_name}",
            title="llm-expect Evaluation Run",
            subtitle=result.run_id
        ))
        
        # Metadata
        grid = Table.grid(padding=1)
        grid.add_column(style="bold")
        grid.add_column()
        grid.add_row("Dataset:", result.dataset_path)
        grid.add_row("Timestamp:", result.timestamp.strftime('%Y-%m-%d %H:%M:%S'))
        grid.add_row("Results Dir:", str(result.run_dir))
        console.print(grid)
        console.print()
        
        # Summary Table
        s = result.summary
        summary_table = Table(title="Summary", show_header=True)
        summary_table.add_column("Metric")
        summary_table.add_column("Value")
        
        summary_table.add_row("Total Tests", str(s.total_tests))
        summary_table.add_row("Passed", f"[green]{s.passed_tests}[/green]")
        summary_table.add_row("Failed", f"[red]{s.failed_tests}[/red]")
        summary_table.add_row("Errors", f"[yellow]{s.error_tests}[/yellow]")
        summary_table.add_row("Success Rate", f"{s.success_rate:.1%}")
        summary_table.add_row("Total Time", f"{s.total_time:.2f}s")
        
        console.print(summary_table)
        
        # Metrics Table
        if s.metrics:
            console.print()
            metrics_table = Table(title="Metrics Breakdown", show_header=True)
            metrics_table.add_column("Metric")
            metrics_table.add_column("Mean Score")
            metrics_table.add_column("Min")
            metrics_table.add_column("Max")
            
            for metric_name, stats in s.metrics.items():
                metrics_table.add_row(
                    metric_name.title(),
                    f"{stats.get('mean', 0):.2f}",
                    f"{stats.get('min', 0):.2f}",
                    f"{stats.get('max', 0):.2f}"
                )
            console.print(metrics_table)
        
        # Failed Tests
        failed_tests = [t for t in result.tests if not t.passed]
        if failed_tests:
            console.print(f"\n[bold red]Failed Tests ({len(failed_tests)})[/bold red]")
            
            for i, test in enumerate(failed_tests):
                if i >= 5:
                    console.print(f"[dim]... and {len(failed_tests) - 5} more[/dim]")
                    break
                    
                error_msg = test.error or "Metric thresholds not met"
                
                # Create a mini-table for the failure
                fail_panel = Panel(
                    f"[bold]Input:[/bold] {test.input}\n"
                    f"[bold]Error:[/bold] [red]{error_msg}[/red]",
                    title=f"Test ID: {test.test_id}",
                    border_style="red"
                )
                console.print(fail_panel)
        
        return 0
        
    except LLMExpectError as e:
        console.print(f"[bold red]Failed to load run:[/bold red] {e}")
        return 1
    except Exception as e:
        console.print(f"[bold red]Unexpected error:[/bold red] {e}")
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
        
    except LLMExpectError as e:
        print(f"❌ Cleanup failed: {e}")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    
    parser = argparse.ArgumentParser(
        prog="llm-expect",
        description="LLM Expect - Evaluation Framework for LLM Functions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  llm-expect runs cleanup --keep 10         # Keep only 10 most recent runs

For more information, visit: https://github.com/osuthorpe/llm-expect
        """
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 0.1.9'
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