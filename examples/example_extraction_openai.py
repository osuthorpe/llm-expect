import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from llm_expect import llm_expect

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@llm_expect(dataset="examples/datasets/extraction.jsonl")
def extract_correct(prompt: str) -> str:
    """Correct implementation: Uses proper system prompt to extract structured JSON."""
    try:
        response = client.chat.completions.create(
            model="gpt-5.1",
            messages=[
                {"role": "system", "content": "Extract the person's information from the text and return it as a JSON object with exactly these keys: name (string), age (integer), city (string). Return ONLY valid JSON, no other text."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content
    except Exception as e:
        return json.dumps({"error": str(e)})

@llm_expect(dataset="examples/datasets/extraction.jsonl")
def extract_incorrect(prompt: str) -> str:
    """Incorrect implementation: Uses bad system prompt that produces narrative text instead of JSON."""
    try:
        response = client.chat.completions.create(
            model="gpt-5.1",
            messages=[
                {"role": "system", "content": "Write a friendly sentence describing the person mentioned in the text."},
                {"role": "user", "content": prompt}
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    print("🚀 Running Extraction Evaluation using OpenAI")
    if client:
        results = extract_correct.run_eval()
        print(f"   Result (Correct): {'✅ PASSED' if results['passed'] else '❌ FAILED'}")
        print(f"   Success Rate: {results['summary']['success_rate']:.1%}")

        results = extract_incorrect.run_eval()
        print(f"   Result (Incorrect): {'✅ PASSED' if results['passed'] else '❌ FAILED'}")
        print(f"   Success Rate: {results['summary']['success_rate']:.1%}")
    else:
        print("   ⚠️ Skipped: OpenAI API key missing or SDK not installed.")
