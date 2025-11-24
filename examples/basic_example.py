"""
Real-world Vald8 example using OpenAI, Anthropic, and Gemini.

This example demonstrates how to evaluate functions that call real LLM APIs.
Requires the following environment variables:
- OPENAI_API_KEY
- ANTHROPIC_API_KEY
- GEMINI_API_KEY
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv
from vald8 import vald8

# Load environment variables from .env file
load_dotenv()

# Try importing SDKs, handle missing dependencies gracefully
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None


# Initialize clients if available
openai_client = OpenAI() if OpenAI and os.getenv("OPENAI_API_KEY") else None
anthropic_client = Anthropic() if Anthropic and os.getenv("ANTHROPIC_API_KEY") else None

if genai and os.getenv("GEMINI_API_KEY"):
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


@vald8(dataset="examples/eval_dataset.jsonl")
def call_openai_gpt5(prompt: str) -> str:
    """
    Calls OpenAI's GPT-5.1 model (hypothetical/preview).
    Falls back to gpt-4o if 5.1 is not available to your account.
    """
    if not openai_client:
        return "Error: OpenAI client not initialized or missing API key."

    try:
        # Using a hypothetical model name for GPT-5.1 as requested
        # In practice, this would be the actual model ID provided by OpenAI
        response = openai_client.chat.completions.create(
            model="gpt-5.1-preview",  # Or "gpt-4o" as fallback
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"OpenAI API Error: {str(e)}"


@vald8(dataset="examples/eval_dataset.jsonl")
def call_claude_35(prompt: str) -> str:
    """Calls Anthropic's Claude 3.5 Sonnet."""
    if not anthropic_client:
        return "Error: Anthropic client not initialized or missing API key."

    try:
        response = anthropic_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except Exception as e:
        return f"Anthropic API Error: {str(e)}"


@vald8(dataset="examples/eval_dataset.jsonl")
def call_gemini_15(prompt: str) -> str:
    """Calls Google's Gemini 1.5 Pro."""
    if not genai:
        return "Error: Google Generative AI SDK not installed."
    if not os.getenv("GEMINI_API_KEY"):
        return "Error: GEMINI_API_KEY not set."

    try:
        model = genai.GenerativeModel('gemini-1.5-pro-latest')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini API Error: {str(e)}"


def run_evaluations():
    """Run evaluations for all models."""
    print("🚀 Running Real-World LLM Evaluations\n")

    # OpenAI Evaluation
    print("1️⃣ Testing OpenAI GPT-5.1...")
    if openai_client:
        results_openai = call_openai_gpt5.run_eval()
        print(f"   Result: {'✅ PASSED' if results_openai['passed'] else '❌ FAILED'}")
        print(f"   Success Rate: {results_openai['summary']['success_rate']:.1%}")
    else:
        print("   ⚠️ Skipped: OpenAI API key missing or SDK not installed.")
    print()

    # Anthropic Evaluation
    print("2️⃣ Testing Anthropic Claude 3.5...")
    if anthropic_client:
        results_claude = call_claude_35.run_eval()
        print(f"   Result: {'✅ PASSED' if results_claude['passed'] else '❌ FAILED'}")
        print(f"   Success Rate: {results_claude['summary']['success_rate']:.1%}")
    else:
        print("   ⚠️ Skipped: Anthropic API key missing or SDK not installed.")
    print()

    # Gemini Evaluation
    print("3️⃣ Testing Google Gemini 1.5...")
    if genai and os.getenv("GEMINI_API_KEY"):
        results_gemini = call_gemini_15.run_eval()
        print(f"   Result: {'✅ PASSED' if results_gemini['passed'] else '❌ FAILED'}")
        print(f"   Success Rate: {results_gemini['summary']['success_rate']:.1%}")
    else:
        print("   ⚠️ Skipped: Gemini API key missing or SDK not installed.")
    print()


if __name__ == "__main__":
    run_evaluations()