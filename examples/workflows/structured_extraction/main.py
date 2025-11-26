import os
import json
from typing import Dict, Any

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

from llm_expect import llm_expect

# Initialize client (optional)
api_key = os.environ.get("ANTHROPIC_API_KEY")
client = Anthropic(api_key=api_key) if (api_key and Anthropic) else None


def mock_extraction(text: str) -> str:
    """Mock response for when API key is missing."""
    if "Alice" in text:
        return '{"name": "Alice", "age": 30, "city": "New York"}'
    if "Bob" in text:
        return '{"name": "Bob", "age": 25, "city": "London"}'
    return '{"error": "No entities found"}'


def extract_user_info(text: str) -> Dict[str, Any]:
    """
    Extracts user information (name, age, city) from text.
    Includes basic retry logic for malformed JSON.
    """
    system_prompt = "You are a data extraction assistant. Extract the user's name, age, and city into a JSON object."
    
    # Simple retry loop (1 retry)
    for attempt in range(2):
        try:
            if client:
                message = client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=1000,
                    system=system_prompt,
                    messages=[{"role": "user", "content": text}]
                )
                content = message.content[0].text
            else:
                content = mock_extraction(text)
            
            # Attempt to parse JSON to validate it before returning
            # This acts as our "runtime validation"
            data = json.loads(content)
            return data
            
        except json.JSONDecodeError:
            if attempt == 0:
                print(f"Attempt {attempt+1} failed: Invalid JSON. Retrying...")
                continue # Retry
            else:
                return {"error": "Failed to extract valid JSON"}
                
    return {"error": "Unknown error"}


# JSONL dataset with test cases
# We use absolute path to allow running from anywhere
dataset_path = os.path.join(os.path.dirname(__file__), "1_extraction_tests.jsonl")


@llm_expect(dataset=os.path.join(os.path.dirname(__file__), "dataset.jsonl"))
def run_extraction_pipeline(text: str) -> Dict[str, Any]:
    """Adapter for JSONL test cases."""
    return extract_user_info(text)


if __name__ == "__main__":
    print(f"Running extraction tests from {dataset_path}...")
    run_extraction_pipeline.run_eval()
