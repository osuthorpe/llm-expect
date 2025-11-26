import os
from dotenv import load_dotenv
from openai import OpenAI
from llm_expect import llm_expect

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("OPENAI_API_KEY") else None

@llm_expect(dataset="examples/datasets/regex.jsonl")
def regex_correct(prompt: str) -> str:
    """Correct implementation: Returns properly formatted date."""
    if client is None:
        return "Error: OPENAI_API_KEY not set"

    response = client.chat.completions.create(
        model="gpt-5.1",
        messages=[
            {"role": "system", "content": "Return only a date in YYYY-MM-DD format, nothing else."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

@llm_expect(dataset="examples/datasets/regex.jsonl")
def regex_incorrect(prompt: str) -> str:
    """Incorrect implementation: Returns date in wrong format."""
    if client is None:
        return "Error: OPENAI_API_KEY not set"

    response = client.chat.completions.create(
        model="gpt-5.1",
        messages=[
            {"role": "system", "content": "Return a date in MM/DD/YYYY format."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    print("🚀 Running Regex Evaluation using OpenAI")

    if not os.getenv("OPENAI_API_KEY"):
        print("   ⚠️ Skipped: OPENAI_API_KEY not set.")
    else:
        results = regex_correct.run_eval()
        print(f"   Result (Correct Format): {'✅ PASSED' if results['passed'] else '❌ FAILED'}")
        print(f"   Success Rate: {results['summary']['success_rate']:.1%}")

        results = regex_incorrect.run_eval()
        print(f"   Result (Wrong Format): {'✅ PASSED' if results['passed'] else '❌ FAILED'}")
        print(f"   Success Rate: {results['summary']['success_rate']:.1%}")
