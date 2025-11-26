"""
HTML Reporting module for LLM Expect.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from jinja2 import Template

from .models import EvaluationResult

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Expect Report - {{ result.function_name }}</title>
    <style>
        :root {
            --primary: #4f46e5;
            --success: #10b981;
            --danger: #ef4444;
            --warning: #f59e0b;
            --bg: #f9fafb;
            --card-bg: #ffffff;
            --text: #1f2937;
            --text-light: #6b7280;
            --border: #e5e7eb;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg);
            color: var(--text);
            line-height: 1.5;
            margin: 0;
            padding: 2rem;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        .header {
            background-color: var(--card-bg);
            padding: 1.5rem;
            border-radius: 0.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .title h1 {
            margin: 0;
            font-size: 1.5rem;
        }
        
        .meta {
            color: var(--text-light);
            font-size: 0.875rem;
        }
        
        .status-badge {
            padding: 0.5rem 1rem;
            border-radius: 9999px;
            font-weight: 600;
            font-size: 0.875rem;
        }
        
        .status-passed {
            background-color: #d1fae5;
            color: #065f46;
        }
        
        .status-failed {
            background-color: #fee2e2;
            color: #991b1b;
        }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }
        
        .card {
            background-color: var(--card-bg);
            padding: 1.5rem;
            border-radius: 0.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        .card h2 {
            margin-top: 0;
            font-size: 1.125rem;
            color: var(--text-light);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-size: 0.75rem;
        }
        
        .stat-value {
            font-size: 2rem;
            font-weight: 700;
            color: var(--primary);
        }
        
        .stat-label {
            color: var(--text-light);
            font-size: 0.875rem;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
        }
        
        th, td {
            padding: 0.75rem 1rem;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }
        
        th {
            background-color: #f3f4f6;
            font-weight: 600;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        tr:last-child td {
            border-bottom: none;
        }
        
        .test-row:hover {
            background-color: #f9fafb;
        }
        
        .test-status {
            display: inline-block;
            width: 0.75rem;
            height: 0.75rem;
            border-radius: 50%;
        }
        
        .test-passed { background-color: var(--success); }
        .test-failed { background-color: var(--danger); }
        
        .error-msg {
            color: var(--danger);
            font-family: monospace;
            font-size: 0.875rem;
            background-color: #fee2e2;
            padding: 0.25rem 0.5rem;
            border-radius: 0.25rem;
        }
        
        .details-row {
            background-color: #f9fafb;
            display: none;
        }
        
        .details-content {
            padding: 1rem;
        }
        
        pre {
            background-color: #1f2937;
            color: #f3f4f6;
            padding: 1rem;
            border-radius: 0.5rem;
            overflow-x: auto;
            font-size: 0.875rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="title">
                <h1>LLM Expect Report: {{ result.function_name }}</h1>
                <div class="meta">
                    Run ID: {{ result.run_id }} • {{ result.timestamp.strftime('%Y-%m-%d %H:%M:%S') }}
                </div>
            </div>
            <div class="status-badge {{ 'status-passed' if result.passed else 'status-failed' }}">
                {{ 'PASSED' if result.passed else 'FAILED' }}
            </div>
        </div>
        
        <div class="grid">
            <div class="card">
                <h2>Success Rate</h2>
                <div class="stat-value">{{ "%.1f"|format(result.summary.success_rate * 100) }}%</div>
                <div class="stat-label">{{ result.summary.passed_tests }} / {{ result.summary.total_tests }} passed</div>
            </div>
            <div class="card">
                <h2>Duration</h2>
                <div class="stat-value">{{ "%.2f"|format(result.summary.total_time) }}s</div>
                <div class="stat-label">Total execution time</div>
            </div>
            <div class="card">
                <h2>Dataset</h2>
                <div class="stat-value" style="font-size: 1.25rem; word-break: break-all;">{{ result.dataset_path }}</div>
            </div>
        </div>
        
        <div class="card">
            <h2>Test Results</h2>
            <table>
                <thead>
                    <tr>
                        <th style="width: 50px">Status</th>
                        <th>ID</th>
                        <th>Input</th>
                        <th>Result/Error</th>
                        <th>Time</th>
                    </tr>
                </thead>
                <tbody>
                    {% for test in result.tests %}
                    <tr class="test-row">
                        <td>
                            <span class="test-status {{ 'test-passed' if test.passed else 'test-failed' }}"></span>
                        </td>
                        <td><strong>{{ test.test_id }}</strong></td>
                        <td>{{ test.input }}</td>
                        <td>
                            {% if test.passed %}
                                <span style="color: var(--success)">Passed</span>
                            {% else %}
                                <div class="error-msg">{{ test.error or "Metric failure" }}</div>
                            {% endif %}
                        </td>
                        <td>{{ "%.3f"|format(test.execution_time) }}s</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
"""

class HTMLReporter:
    """Generates HTML reports for evaluation results."""
    
    def generate_report(self, result: EvaluationResult) -> str:
        """
        Generate HTML report string.
        
        Args:
            result: Evaluation result object
            
        Returns:
            HTML string
        """
        template = Template(HTML_TEMPLATE)
        return template.render(result=result)
    
    def save_report(self, result: EvaluationResult, output_path: Path) -> None:
        """
        Generate and save HTML report to file.
        
        Args:
            result: Evaluation result object
            output_path: Path to save HTML file
        """
        html_content = self.generate_report(result)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
