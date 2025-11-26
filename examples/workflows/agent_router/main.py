import os
import json
from typing import Dict, Any, List

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

from llm_expect import llm_expect

# Initialize client (optional)
api_key = os.environ.get("ANTHROPIC_API_KEY")
client = Anthropic(api_key=api_key) if (api_key and Anthropic) else None

TOOLS = [
    {"name": "weather_lookup", "description": "Get current weather for a city"},
    {"name": "calculator", "description": "Perform mathematical calculations"},
    {"name": "web_search", "description": "Search the internet for current events"}
]


def mock_router(query: str) -> str:
    """Mock router logic for demonstration."""
    if "weather" in query: return '{"tool": "weather_lookup", "args": {"city": "Paris"}}'
    if "plus" in query or "+" in query: return '{"tool": "calculator", "args": {"expression": "2+2"}}'
    return '{"tool": "web_search", "args": {"query": "..."}}'


def route_query(user_query: str) -> Dict[str, Any]:
    """
    Decides which tool to use for a given user query.
    Returns a JSON object with 'tool' and 'args'.
    """
    system_prompt = f"""
    You are a routing agent. Available tools: {json.dumps(TOOLS)}.
    Output strictly JSON: {{"tool": "tool_name", "args": {{...}}}}
    """
    
    if not client:
        content = mock_router(user_query)
    else:
        msg = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=200,
            system=system_prompt,
            messages=[{"role": "user", "content": user_query}]
        )
        content = msg.content[0].text
    
    try:
        return json.loads(content)
    except:
        return {"error": "Failed to parse router output", "raw": content}


# JSONL dataset with test cases
# We use absolute path to allow running from anywhere
dataset_path = os.path.join(os.path.dirname(__file__), "3_router_tests.jsonl")


@llm_expect(dataset=os.path.join(os.path.dirname(__file__), "dataset.jsonl"))
def test_router(query: str) -> Dict[str, Any]:
    """Adapter for JSONL test cases."""
    return route_query(query)


if __name__ == "__main__":
    print(f"Running router tests from {dataset_path}...")
    test_router.run_eval()
