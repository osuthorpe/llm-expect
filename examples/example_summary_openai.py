import os
from dotenv import load_dotenv
from openai import OpenAI
from vald8 import vald8

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@vald8(dataset="examples/datasets/summary.jsonl")
def summarize_correct(prompt: str) -> str:
    """Correct implementation: Summarizes the text."""
    try:
        response = client.chat.completions.create(
            model="gpt-5.1",
            messages=[
                {"role": "system", "content": "Summarize the following text, ensuring key entities like Apollo 11, Neil Armstrong, and the date are included."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

@vald8(dataset="examples/datasets/summary.jsonl")
def summarize_incorrect(prompt: str) -> str:
    """Incorrect implementation: Returns irrelevant text."""
    return "I don't know about space missions."

if __name__ == "__main__":
    print("Running Summarization Examples...")
    # The @vald8 decorator handles execution when the script is run
