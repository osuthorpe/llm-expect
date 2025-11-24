import os
from dotenv import load_dotenv
from openai import OpenAI
from vald8 import vald8
from vald8.judges import OpenAIJudgeProvider

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Configure the judge provider
judge_provider = OpenAIJudgeProvider(api_key=api_key, model="gpt-4o")

@vald8(dataset="examples/datasets/judge.jsonl", judge_provider=judge_provider)
def judge_correct(prompt: str) -> str:
    """Correct implementation: Returns a polite email."""
    return "Dear Hiring Manager,\n\nThank you so much for the offer. I am honored, but I must respectfully decline at this time.\n\nBest regards,\n[Name]"

@vald8(dataset="examples/datasets/judge.jsonl", judge_provider=judge_provider)
def judge_incorrect(prompt: str) -> str:
    """Incorrect implementation: Returns a rude email."""
    return "No thanks. Bye."

if __name__ == "__main__":
    print("Running Judge Examples...")
