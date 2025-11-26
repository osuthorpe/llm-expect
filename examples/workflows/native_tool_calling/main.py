import os
import json
from typing import Dict, Any, List

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from llm_expect import llm_expect

# Initialize client (optional)
api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if (api_key and OpenAI) else None

# Define the tool schema (OpenAI format)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get the current stock price for a given symbol",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "The stock symbol, e.g. AAPL",
                    },
                    "currency": {
                        "type": "string",
                        "enum": ["USD", "EUR"],
                        "default": "USD",
                    }
                },
                "required": ["symbol"],
            },
        },
    }
]


def mock_tool_call(prompt: str) -> Dict[str, Any]:
    """Mock response for when OpenAI is not available."""
    if "AAPL" in prompt or "Apple" in prompt:
        return {"name": "get_stock_price", "arguments": {"symbol": "AAPL", "currency": "USD"}}
    if "Google" in prompt:
        return {"name": "get_stock_price", "arguments": {"symbol": "GOOGL", "currency": "USD"}}
    return {"error": "No tool called"}


def get_tool_call(user_prompt: str) -> Dict[str, Any]:
    """
    Sends the prompt to the LLM with tools enabled.
    Returns the name and arguments of the first tool call.
    """
    if not client:
        return mock_tool_call(user_prompt)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": user_prompt}],
        tools=TOOLS,
        tool_choice="auto",
    )

    message = response.choices[0].message
    
    if message.tool_calls:
        tool_call = message.tool_calls[0]
        return {
            "name": tool_call.function.name,
            "arguments": json.loads(tool_call.function.arguments)
        }
    
    return {"error": "No tool called"}


# JSONL dataset with test cases
dataset_path = os.path.join(os.path.dirname(__file__), "5_tool_tests.jsonl")


@llm_expect(dataset=os.path.join(os.path.dirname(__file__), "dataset.jsonl"))
def test_tool_selection(prompt: str) -> Dict[str, Any]:
    """Adapter for JSONL test cases."""
    return get_tool_call(prompt)


if __name__ == "__main__":
    print(f"Running tool calling tests from {dataset_path}...")
    test_tool_selection.run_eval()
