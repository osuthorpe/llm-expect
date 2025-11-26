import os
from typing import Dict, Any

try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_openai import ChatOpenAI
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

from llm_expect import llm_expect

# Initialize LangChain components if available
if LANGCHAIN_AVAILABLE:
    # We use a simple chain: Prompt -> LLM -> String Parser
    prompt = ChatPromptTemplate.from_template("Tell me a short joke about {topic}.")
    model = ChatOpenAI(model="gpt-3.5-turbo", api_key=os.environ.get("OPENAI_API_KEY", "mock-key"))
    output_parser = StrOutputParser()
    
    chain = prompt | model | output_parser
else:
    chain = None


def mock_chain_invoke(topic: str) -> str:
    """Mock response for when LangChain is not installed."""
    if "chicken" in topic.lower():
        return "Why did the chicken cross the road? To get to the other side!"
    return f"Here is a funny joke about {topic}."


def generate_joke(topic: str) -> str:
    """
    Generates a joke using a LangChain LCEL chain.
    """
    if not LANGCHAIN_AVAILABLE or not os.environ.get("OPENAI_API_KEY"):
        return mock_chain_invoke(topic)
        
    try:
        return chain.invoke({"topic": topic})
    except Exception as e:
        return f"Error invoking chain: {str(e)}"


# JSONL dataset with test cases
dataset_path = os.path.join(os.path.dirname(__file__), "4_langchain_tests.jsonl")


@llm_expect(dataset=os.path.join(os.path.dirname(__file__), "dataset.jsonl"))
def run_langchain_pipeline(topic: str) -> str:
    """Adapter for JSONL test cases."""
    return generate_joke(topic)


if __name__ == "__main__":
    print(f"Running LangChain tests from {dataset_path}...")
    run_langchain_pipeline.run_eval()
