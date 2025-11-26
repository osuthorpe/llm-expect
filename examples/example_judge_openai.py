import os
from dotenv import load_dotenv
from openai import OpenAI
from llm_expect import llm_expect

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Example model client used by the functions under test.
# JudgeProvider uses its own HTTP client internally.
client = OpenAI(api_key=api_key) if api_key else None

@llm_expect(
    dataset="examples/datasets/judge.jsonl",
    tests=["custom_judge"],
    judge_provider="openai",
    judge_model="gpt-5.1",
)
def judge_correct(prompt: str) -> str:
    """Correct implementation: Returns a polite, professional email."""
    if client is None:
        # In real usage you might raise instead.
        return "Error: OPENAI_API_KEY not set"

    response = client.chat.completions.create(
        model="gpt-5.1",
        messages=[
            {"role": "system", "content": "Write a polite and professional email."},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


@vald8(
    dataset="examples/datasets/judge.jsonl",
    tests=["custom_judge"],
    judge_provider="openai",
    judge_model="gpt-5.1",
)
def judge_incorrect(prompt: str) -> str:
    """Incorrect implementation: Returns a rude, unprofessional email."""
    if client is None:
        return "Error: OPENAI_API_KEY not set"

    response = client.chat.completions.create(
        model="gpt-5.1",
        messages=[
            {
                "role": "system",
                "content": "Write a very brief, casual, and somewhat rude response.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    print("🚀 Running Judge Evaluation using OpenAI")

    if not api_key:
        print("   ⚠️ Skipped: OPENAI_API_KEY not set.")
    else:
        results = judge_correct.run_eval()
        print(f"   Result (Polite): {'✅ PASSED' if results['passed'] else '❌ FAILED'}")
        print(f"   Success Rate: {results['summary']['success_rate']:.1%}")

        results = judge_incorrect.run_eval()
        print(f"   Result (Rude): {'✅ PASSED' if results['passed'] else '❌ FAILED'}")
        print(f"   Success Rate: {results['summary']['success_rate']:.1%}")