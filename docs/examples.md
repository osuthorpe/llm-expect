# Advanced Examples

This page details realistic, complex workflows using LLM Expect.
You can find the full source code for these examples in the `examples/workflows/` directory.

## 1. Structured Extraction with Retry Logic
**File:** `examples/workflows/structured_extraction/main.py`

This example demonstrates how to:
*   Extract structured JSON data (Name, Age, City) from unstructured text.
*   Implement a retry loop that catches `json.JSONDecodeError`.
*   Use the `schema` metric to validate the output structure.

```python
@llm_expect(dataset="1_extraction_tests.jsonl")
def run_extraction_pipeline(text: str) -> Dict[str, Any]:
    # Tries to extract JSON, retries once on failure
    return extract_user_info(text)
```

**Key Dataset Pattern:**
```json
{
  "expected": {
    "schema": {
      "required": ["name", "age", "city"],
      "properties": {"age": {"type": "integer"}}
    }
  }
}
```

## 2. Multi-Step Chain (Summarize -> Translate)
**File:** `examples/workflows/multi_step_chain/main.py`

This example shows how to test a pipeline where the output of one LLM call feeds into another.
*   **Step 1:** Summarize a long text.
*   **Step 2:** Translate the summary to Spanish.
*   **Testing:** We return a dictionary containing both the intermediate summary and the final translation to test them together.

```python
def processing_chain(text: str) -> Dict[str, str]:
    summary = summarize_text(text)
    translation = translate_text(summary, "Spanish")
    return {"original_summary": summary, "translation": translation}
```

**Key Dataset Pattern:**
```json
{
  "expected": {
    "contains": ["energía"],  # Check translation content
    "judge": {
      "prompt": "Is 'translation' a valid Spanish translation of 'original_summary'?"
    }
  }
}
```

## 5. Native Tool Calling (OpenAI)
**File:** `examples/workflows/native_tool_calling/main.py`

This example tests the "brain" of an agent—the router that decides which tool to call.
*   **Scenario:** A user asks a question, and the LLM must output a JSON tool call.
*   **Testing:** We verify that the correct tool is selected for a given query.

```python
def route_query(user_query: str) -> Dict[str, Any]:
    # Returns {"tool": "weather", "args": {...}}
    return router_llm(user_query)
```

**Key Dataset Pattern:**
```json
{
  "input": "What is 55 + 10?",
  "expected": {
    "schema": {
      "properties": {
        "tool": {"const": "calculator"}
      }
    }
  }
}
```

## 4. LangChain Integration
**File:** `examples/workflows/langchain_integration/main.py`

LLM Expect works perfectly with LangChain (LCEL). You simply wrap the `chain.invoke()` call in your decorated function.

```python
# Define your chain
chain = prompt | model | output_parser

@llm_expect(dataset="4_langchain_tests.jsonl")
def run_chain(topic: str) -> str:
    # Adapt the input string to the dictionary expected by the chain
    return chain.invoke({"topic": topic})
```

**Key Dataset Pattern:**
```json
{
  "input": "chicken",
  "expected": {
    "contains": ["cross the road"],
    "judge": {"prompt": "Is this a funny joke?"}
  }
}
```
