# Design Philosophy

LLM Expect was built with a specific worldview on how LLM evaluation should work. We believe that evaluation should be **local**, **deterministic**, and **code-centric**.

## Core Principles

### 1. Local-First & Privacy-Centric
We do not send your data to our servers.
*   **No Login Required:** You don't need an account to use LLM Expect.
*   **No Cloud Dashboard:** Your results live on your machine (or your CI runner).
*   **Your Keys, Your Control:** You manage your own API keys. We never see them.

### 2. Zero-Config by Default
You shouldn't need a 50-line YAML file to run a test.
*   **Sensible Defaults:** We assume you want to test accuracy and safety unless you say otherwise.
*   **Convention over Configuration:** If you name your dataset `tests.jsonl`, we'll find it.

### 3. Code-Based Testing (Not UI-Based)
Evaluation belongs in your codebase, version-controlled alongside your application logic.
*   **Git-Friendly:** JSONL datasets and Python test files are easy to diff and review.
*   **CI/CD Native:** Since it's just a Python script, it runs anywhere Python runs.

### 4. Minimal Surface Area
We focus on doing one thing well: **running a function against a dataset and checking the output.**
*   We are *not* an agent framework.
*   We are *not* a prompt management tool.
*   We are *not* a vector database.

## Comparison to Other Tools

| Feature | LLM Expect | DeepEval / Ragas | LangSmith / Arize |
| :--- | :--- | :--- | :--- |
| **Primary Interface** | Python Decorator | Python SDK | Web Dashboard |
| **Data Storage** | Local JSONL | Local / Cloud | Cloud |
| **Focus** | Integration Testing | RAG Metrics | Observability |
| **Complexity** | Low | High | High |
| **Cost** | Free (Open Source) | Free / Paid | Paid |

## Why "Expect"?
The name comes from the testing assertion pattern (e.g., `expect(result).toBe(value)`). We want LLM testing to feel as rigorous and standard as unit testing.
