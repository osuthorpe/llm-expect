import os
from dotenv import load_dotenv
from openai import OpenAI
from llm_expect import llm_expect

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("OPENAI_API_KEY") else None

@llm_expect(dataset="examples/datasets/reference.jsonl")
def reference_correct(prompt: str) -> str:
    """Correct implementation: Returns exact expected answer."""
    if client is None:
        return "Error: OPENAI_API_KEY not set"

    response = client.chat.completions.create(
        model="gpt-5.1",
        messages=[
            {"role": "system", "content": "Answer with only the number, no explanation."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

@llm_expect(dataset="examples/datasets/reference.jsonl")
def reference_incorrect(prompt: str) -> str:
    """Incorrect implementation: Returns wrong answer."""
    if client is None:
        return "Error: OPENAI_API_KEY not set"

    response = client.chat.completions.create(
        model="gpt-5.1",
        messages=[
            {"role": "system", "content": "Answer the math question but add 1 to the result."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

if __name__ == "__main__":
    print("🚀 Running Reference Evaluation using OpenAI")

    if not os.getenv("OPENAI_API_KEY"):
        print("   ⚠️ Skipped: OPENAI_API_KEY not set.")
    else:
        results = reference_correct.run_eval()
        print(f"   Result (Correct Answer): {'✅ PASSED' if results['passed'] else '❌ FAILED'}")
        print(f"   Success Rate: {results['summary']['success_rate']:.1%}")

        results = reference_incorrect.run_eval()
        print(f"   Result (Wrong Answer): {'✅ PASSED' if results['passed'] else '❌ FAILED'}")
        print(f"   Success Rate: {results['summary']['success_rate']:.1%}")
