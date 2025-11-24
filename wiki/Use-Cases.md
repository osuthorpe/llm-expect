# Use Cases

Vald8 is flexible enough to handle a wide variety of LLM evaluation scenarios. Here are three common patterns.

## 1. RAG (Retrieval-Augmented Generation) Evaluation

**The Problem**: You have a RAG system that answers questions based on documents. You need to ensure the answers are accurate and don't hallucinate.

**The Setup**:
*   **Dataset**: `questions.jsonl` containing `{"input": "...", "expected": {"judge": {"prompt": "Is the answer supported by the context?"}}}`.
*   **Function**: Your RAG pipeline (retrieve -> generate).

**Vald8 Solution**:
Use a **Custom Judge** to evaluate "Faithfulness" and "Answer Relevance".

```python
@vald8(
    dataset="datasets/rag_eval.jsonl",
    metric="judge"
)
def rag_pipeline(query):
    docs = retriever.search(query)
    answer = generator.generate(query, docs)
    return answer
```

## 2. Safety & Refusal Testing

**The Problem**: You need to ensure your chatbot refuses to answer harmful queries (e.g., bomb-making instructions) but *does* answer safe ones.

**The Setup**:
*   **Dataset**: `safety.jsonl` containing harmful prompts.
*   **Expectation**: `{"safety": true}`.

**Vald8 Solution**:
Use the built-in **Safety Metric**. It checks if the model output indicates a refusal (e.g., "I cannot answer that").

```python
@vald8(
    dataset="datasets/safety_attack.jsonl",
    metric="safety"
)
def safe_chatbot(user_input):
    return llm.chat(user_input)
```

## 3. Structured Data Extraction

**The Problem**: You are using an LLM to extract structured data (e.g., JSON) from unstructured text (e.g., emails). You need to verify the output is valid JSON and matches a schema.

**The Setup**:
*   **Dataset**: `emails.jsonl`.
*   **Expectation**: `{"contains": ["order_id", "customer_name"]}` or `{"regex": "Order ID: \\d+"}`.

**Vald8 Solution**:
Use **Regex** or **Contains** metrics for deterministic validation.

```python
@vald8(
    dataset="datasets/emails.jsonl",
    metric="regex"
)
def extract_order_info(email_body):
    # Returns a JSON string
    return extractor.run(email_body)
```

## 4. Regression Testing

**The Problem**: You are changing your prompt or switching models (e.g., GPT-4 to GPT-3.5) and want to make sure quality doesn't drop.

**The Setup**:
1.  Run Vald8 on your "Golden Dataset" with the old model.
2.  Switch the model.
3.  Run Vald8 again.
4.  Compare the pass rates.

**Vald8 Solution**:
Vald8's session-based logging makes this easy. You can simply look at the pass rates in the terminal or inspect the `results.jsonl` files to see exactly which cases regressed.
