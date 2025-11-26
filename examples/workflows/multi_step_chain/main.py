import os
from typing import Dict, Any

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

from llm_expect import llm_expect

# Initialize client (optional)
api_key = os.environ.get("ANTHROPIC_API_KEY")
client = Anthropic(api_key=api_key) if (api_key and Anthropic) else None


def summarize_text(text: str) -> str:
    """Summarize input text in one sentence."""
    if not client:
        # Fallback mock for docs and local runs without an API key
        return "This text discusses the importance of renewable energy."

    msg = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=100,
        messages=[
            {
                "role": "user",
                "content": f"Summarize this in one sentence: {text}",
            }
        ],
    )
    return msg.content[0].text


def translate_text(text: str, target_lang: str) -> str:
    """Translate text to the target language."""
    if not client:
        # Simple mock translation for demonstration
        if target_lang.lower() == "french":
            return "Ce texte traite de l'importance des énergies renouvelables."
        return "Este texto discute la importancia de la energía renovable."

    msg = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=100,
        messages=[
            {
                "role": "user",
                "content": f"Translate this to {target_lang}: {text}",
            }
        ],
    )
    return msg.content[0].text


def processing_chain(text: str, target_lang: str = "Spanish") -> Dict[str, str]:
    """
    A small multi-step chain:
    1. Summarize the input text.
    2. Translate the summary.
    3. Return both.
    """
    summary = summarize_text(text)
    translation = translate_text(summary, target_lang)

    return {
        "original_summary": summary,
        "translation": translation,
        "language": target_lang,
    }


# JSONL dataset with test cases for the chain
# We use absolute path to allow running from anywhere
dataset_path = os.path.join(os.path.dirname(__file__), "2_chain_tests.jsonl")


@llm_expect(dataset=os.path.join(os.path.dirname(__file__), "dataset.jsonl"))
def run_chain(input_data: Dict[str, Any]) -> Dict[str, str]:
    """Adapter for JSONL test cases into the processing chain."""
    return processing_chain(
        input_data["text"],
        input_data.get("target_lang", "Spanish"),
    )


if __name__ == "__main__":
    print(f"Running chain tests from {dataset_path}...")
    run_chain.run_eval()
