# Roadmap

This roadmap outlines the future direction of LLM Expect. We prioritize features that align with our philosophy of simplicity and developer experience.

## 🚀 Upcoming Features

### 1. Dataset Builder UI
A simple, local web interface to visually create and edit `tests.jsonl` files.
*   **Status:** In Progress
*   **Goal:** Reduce friction for creating complex JSONL structures.

### 2. Agent Step Evaluation
Better support for testing intermediate steps in agent chains.
*   **Status:** Planned
*   **Goal:** Allow decorating individual tools or reasoning steps.

### 3. Embedding-Based Scoring
New metric using cosine similarity of embeddings for semantic matching.
*   **Status:** Planned
*   **Goal:** Cheaper and faster than LLM-as-a-judge for semantic similarity.

### 4. Optional CLI Runner
A `llm-expect run` command to execute tests without writing a Python script.
*   **Status:** Under Consideration
*   **Goal:** Simplify CI/CD integration.

## 🔮 Long-Term Vision

*   **IDE Integration:** VS Code extension for running tests directly from the editor.
*   **Custom Reporters:** Plug-and-play reporters for Slack, Discord, or custom webhooks.
*   **Parallel Cloud Execution:** Optional ability to offload execution to a cloud runner for massive datasets.

## ❌ Out of Scope

*   **Prompt Management:** We will not build a prompt registry.
*   **Observability Dashboard:** We will not build a hosted dashboard.
