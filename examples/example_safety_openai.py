import os
from dotenv import load_dotenv
from openai import OpenAI
from vald8 import vald8

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@vald8(dataset="examples/datasets/safety.jsonl")
def safe_wrapper(prompt: str) -> str:
    """Correct implementation: Refuses harmful requests."""
    # Simple guardrail simulation
    harmful_keywords = ["bomb", "weapon", "kill"]
    if any(k in prompt.lower() for k in harmful_keywords):
        return "I cannot help you with that request."
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

@vald8(dataset="examples/datasets/safety.jsonl")
def unsafe_wrapper(prompt: str) -> str:
    """Incorrect implementation: Answers harmful requests (simulated)."""
    # For demonstration, we force a harmful response if the prompt asks for it
    if "bomb" in prompt.lower():
        return "Here is how to build a bomb: [REDACTED]"
    
    # Otherwise just call the model
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    print("Running Safety Examples...")
