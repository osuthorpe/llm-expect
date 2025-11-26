import os
import sys
from llm_expect import llm_expect
from llm_expect.errors import JudgeProviderError

# Ensure API key is unset
if "OPENAI_API_KEY" in os.environ:
    del os.environ["OPENAI_API_KEY"]

print("1. Defining decorated function without API key...")
try:
    # This should NOT raise an error now (lazy init)
    @llm_expect(
        dataset="tests.jsonl", 
        judge_provider="openai",
        tests=["safety"] # Requires judge
    )
    def my_llm_func(prompt):
        return "response"
        
    print("✅ SUCCESS: Import/Definition did not raise error.")
except Exception as e:
    print(f"❌ FAILURE: Import/Definition raised error: {e}")
    sys.exit(1)

print("\n2. Running evaluation (expecting failure due to missing key)...")
try:
    # This SHOULD raise an error when it tries to init the judge
    my_llm_func.run_eval()
    print("❌ FAILURE: run_eval() did not raise expected error.")
    sys.exit(1)
except Exception as e:
    # We expect a JudgeProviderError or similar wrapping it
    print(f"✅ SUCCESS: run_eval() raised expected error: {e}")
    sys.exit(0)
