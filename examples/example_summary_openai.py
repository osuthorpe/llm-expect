import os
from dotenv import load_dotenv
from openai import OpenAI
from llm_expect import llm_expect

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@llm_expect(dataset="examples/datasets/summary.jsonl")
def summarize_correct(prompt: str) -> str:
    """Correct implementation: Summarizes the text with key entities."""
    try:
        response = client.chat.completions.create(
            model="gpt-5.1",
            messages=[
                {"role": "system", "content": "Summarize the following text, ensuring key entities like Apollo 11, Neil Armstrong, Buzz Aldrin, and the date 1969 are included."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

@llm_expect(dataset="examples/datasets/summary.jsonl")
def summarize_incorrect(prompt: str) -> str:
    """Incorrect implementation: Returns irrelevant summary that misses key entities."""
    try:
        response = client.chat.completions.create(
            model="gpt-5.1",
            messages=[
                {"role": "system", "content": "Write a one-sentence summary about space exploration in general."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    print("🚀 Running Summarization Evaluation using OpenAI")
    if client:
        results = summarize_correct.run_eval()
        print(f"   Result (Correct): {'✅ PASSED' if results['passed'] else '❌ FAILED'}")
        print(f"   Success Rate: {results['summary']['success_rate']:.1%}")

        results = summarize_incorrect.run_eval()
        print(f"   Result (Incorrect): {'✅ PASSED' if results['passed'] else '❌ FAILED'}")
        print(f"   Success Rate: {results['summary']['success_rate']:.1%}")
    else:
        print("   ⚠️ Skipped: OpenAI API key missing or SDK not installed.")
