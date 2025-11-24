from vald8 import vald8
import json
import os

# Create a dummy dataset
with open("repro_tests.jsonl", "w") as f:
    f.write(json.dumps({"id": "math1", "input": "What is 2+2?", "expected": {"reference": "4"}}) + "\n")
    f.write(json.dumps({"id": "json1", "input": "Return JSON with name and age", "expected": {"schema": {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "number"}}, "required": ["name", "age"]}}}) + "\n")
    f.write(json.dumps({"id": "hello1", "input": "Greet politely", "expected": {"contains": ["hello", "please"]}}) + "\n")
    f.write(json.dumps({"id": "regex1", "input": "Give a date", "expected": {"regex": r"\d{4}-\d{2}-\d{2}"}}) + "\n")

# Mock LLM function
@vald8(dataset="repro_tests.jsonl")
def generate(prompt: str) -> dict:
    if "2+2" in prompt:
        return "4"
    elif "JSON" in prompt:
        return json.dumps({"name": "Alice", "age": 30})
    elif "Greet" in prompt:
        return "Hello there, could you please help me?"
    elif "date" in prompt:
        return "2023-10-27"
    return "Unknown"

if __name__ == "__main__":
    results = generate.run_eval()
    print(f"Passed: {results['passed']}")
    print(f"Summary: {json.dumps(results['summary'], indent=2)}")
    
    # Print details of failed tests
    if not results['passed']:
        print("\nFailed Tests:")
        with open(os.path.join(results['run_dir'], "results.jsonl"), "r") as f:
            for line in f:
                test_result = json.loads(line)
                if not test_result['passed']:
                    print(json.dumps(test_result, indent=2))

    # Clean up
    if os.path.exists("repro_tests.jsonl"):
        os.remove("repro_tests.jsonl")
