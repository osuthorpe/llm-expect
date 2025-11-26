# Why JSONL?

LLM Expect uses **JSON Lines (JSONL)** as its primary data format. You might wonder: why not CSV, YAML, or a database?

## 1. Git-Friendly Diffing
JSONL places one record per line. This makes `git diff` extremely readable.
*   **CSV:** Changing a comma or quote can shift columns and make diffs messy.
*   **JSON:** A single change can affect multiple lines (brackets, commas).
*   **JSONL:** Adding or removing a test case affects exactly one line.

## 2. Streaming & Scalability
JSONL can be read line-by-line without loading the entire file into memory.
*   This allows LLM Expect to handle datasets with thousands of examples efficiently.
*   If a test run crashes, you can resume from the last processed line easily.

## 3. Rich Structure
Unlike CSV, JSONL supports nested objects.
*   We can store complex `expected` criteria (dictionaries, lists) naturally.
*   We can store chat history (list of messages) without awkward escaping.

## 4. Universal Standard
JSONL is the standard format for:
*   OpenAI Fine-tuning
*   Anthropic Batch API
*   Hugging Face Datasets

By using JSONL, your test data is ready to be used for fine-tuning or other ML workflows without conversion.
