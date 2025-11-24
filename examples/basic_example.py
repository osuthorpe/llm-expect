"""
Basic Vald8 usage example.

Shows how to use the @vald8 decorator for simple LLM function evaluation.
"""


from vald8 import vald8


# Example 1: Simple text generation function
@vald8(dataset="examples/basic_tests.jsonl")
def simple_responder(prompt: str) -> str:
    """A simple responder that handles basic questions."""
    
    # Simple rule-based responses for demo
    if "hello" in prompt.lower():
        return "Hello! How can I help you?"
    elif "name" in prompt.lower():
        return "I'm a helpful AI assistant."
    elif "2+2" in prompt:
        return "4"
    elif "capital of france" in prompt.lower():
        return "The capital of France is Paris."
    else:
        return "I'm not sure about that."


# Example 2: Structured JSON response function
@vald8(
    dataset="examples/structured_tests.jsonl",
    tests=["schema_fidelity", "accuracy"]
)
def extract_person_info(text: str) -> dict:
    """Extract person information and return structured JSON."""
    
    # Simple extraction logic for demo
    result = {"name": None, "age": None, "occupation": None}
    
    if "John" in text:
        result["name"] = "John"
    if "30" in text:
        result["age"] = 30
    if "engineer" in text.lower():
        result["occupation"] = "engineer"
    
    return result


# Example 3: Multi-parameter function
@vald8(dataset="examples/multi_param_tests.jsonl")
def generate_greeting(name: str, time_of_day: str) -> str:
    """Generate personalized greetings."""
    
    if time_of_day.lower() == "morning":
        return f"Good morning, {name}! Have a great day!"
    elif time_of_day.lower() == "evening":
        return f"Good evening, {name}! Hope you had a nice day!"
    else:
        return f"Hello, {name}!"






def run_examples():
    """Run all the example evaluations."""
    
    print("🚀 Running Vald8 Examples\n")
    
    # Example 1: Basic text responses
    print("1️⃣ Testing Simple Responder...")
    results1 = simple_responder.run_eval()
    print(f"   Result: {'✅ PASSED' if results1['passed'] else '❌ FAILED'}")
    print(f"   Success Rate: {results1['summary']['success_rate']:.1%}")
    print(f"   Tests: {results1['summary']['passed_tests']}/{results1['summary']['total_tests']}")
    print()
    
    # Example 2: Structured responses with schema validation
    print("2️⃣ Testing Person Info Extraction...")
    results2 = extract_person_info.run_eval()
    print(f"   Result: {'✅ PASSED' if results2['passed'] else '❌ FAILED'}")
    print(f"   Success Rate: {results2['summary']['success_rate']:.1%}")
    print(f"   Tests: {results2['summary']['passed_tests']}/{results2['summary']['total_tests']}")
    print()
    
    # Example 3: Multi-parameter functions
    print("3️⃣ Testing Greeting Generation...")
    results3 = generate_greeting.run_eval()
    print(f"   Result: {'✅ PASSED' if results3['passed'] else '❌ FAILED'}")
    print(f"   Success Rate: {results3['summary']['success_rate']:.1%}")
    print(f"   Tests: {results3['summary']['passed_tests']}/{results3['summary']['total_tests']}")
    print()
    
    # Show example of normal function usage
    print("📝 Normal Function Usage Examples:")
    print(f'   simple_responder("Hello!"): "{simple_responder("Hello!")}"')
    print(f'   extract_person_info("John is 30"): {extract_person_info("John is 30")}')
    print(f'   generate_greeting("Alice", "morning"): "{generate_greeting("Alice", "morning")}"')
    print()
    
    print("🎉 All examples completed!")
    print(f"📁 Results saved in: {results1['run_dir']}")


if __name__ == "__main__":
    run_examples()