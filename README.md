# 📦 Vald8 SDK

Vald8 is a lightweight Python SDK for **automated LLM evaluation**.  
Run structured tests against your LLM functions with JSONL datasets, validate behavior, and prevent silent regressions — locally or in CI/CD.

✅ **Works fully local & offline** — no external dependencies required  
✅ **Configurable LLM-as-Judge** — OpenAI, Anthropic, AWS Bedrock, Local  
✅ **Developer-first, CI-friendly** — simple decorators, clear outputs  

---

## 🚀 What is Vald8?

**Think of Vald8 like unit tests, but for your AI functions.**

If you're building with LLMs (like ChatGPT, Claude, etc.), you probably have functions that send prompts and get responses. But how do you know if they're working well? 

Vald8 helps you:
1. **Test your AI functions automatically** - No manual checking needed
2. **Catch problems early** - Before your users see bad responses  
3. **Track improvements** - See if changes make things better or worse

**How it works:**
1. Add `@vald8` to any function that calls an AI model
2. Create a simple test file with examples 
3. Run tests to see how well your AI is performing

That's it! Vald8 handles the rest.

---

## ✨ What Can Vald8 Test?

**🎯 Accuracy** - Does your AI give the right answers?
- Check if responses match expected answers
- See if responses contain key information
- Perfect for Q&A, fact-checking, data extraction

**🛡️ Safety** - Is your AI saying appropriate things?  
- Detect harmful or inappropriate content
- Check for bias or problematic responses
- Keep your AI family-friendly and professional

**📋 Following Instructions** - Does your AI do what you asked?
- Check if AI follows your prompts correctly  
- Verify responses match the requested format
- Ensure AI stays on topic

**📊 Format Checking** - Is the output structured correctly?
- Validate JSON responses have the right fields
- Check if data types are correct
- Ensure consistent formatting

**⚡ Works Everywhere**
- No API keys required (but supports OpenAI, Claude, etc. for advanced features)
- Run locally on your machine
- Easy to add to automated testing  

---

## 📝 How to Create Test Data

**Test data is just a simple text file with examples.**

Each line has one test case in this format:
```json
{"id": "test1", "input": "your prompt here", "expected": {"reference": "expected answer"}}
```

**Example test file (save as `my_tests.jsonl`):**
```json
{"id": "math1", "input": "What is 2 + 2?", "expected": {"reference": "4"}}
{"id": "math2", "input": "What is 5 + 3?", "expected": {"reference": "8"}}
{"id": "greeting", "input": "Say hello", "expected": {"contains": ["hello", "hi"]}}
```

**What each part means:**
- **`id`** - A unique name for this test (like "math1", "greeting")
- **`input`** - What you want to send to your AI function  
- **`expected`** - What kind of response you want back

**Types of expected responses:**
- **`"reference": "exact answer"`** - AI must give exactly this answer
- **`"contains": ["word1", "word2"]`** - AI response must include these words
- **`"regex": "pattern"`** - For advanced users who know regular expressions

---

## 🚀 Quick Start - Your First AI Test

### Step 1: Install Vald8
```bash
pip install vald8
```

### Step 2: Write a Simple AI Function
```python
# my_app.py
import openai
from vald8 import vald8

# Your AI function (this is what you want to test)
def ask_ai(question):
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": question}]
    )
    return response.choices[0].message.content

# Add the @vald8 decorator to test it
@vald8(dataset="my_tests.jsonl", tests=["accuracy"])
def ask_ai_with_tests(question):
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo", 
        messages=[{"role": "user", "content": question}]
    )
    return response.choices[0].message.content
```

### Step 3: Create Your Test File
Create a file called `my_tests.jsonl`:
```json
{"id": "math1", "input": "What is 2 + 2?", "expected": {"reference": "4"}}
{"id": "capital", "input": "What is the capital of France?", "expected": {"contains": ["Paris"]}}
{"id": "greeting", "input": "Say hello nicely", "expected": {"contains": ["hello"]}}
```

### Step 4: Run Your Tests
```python
# Use your function normally
answer = ask_ai_with_tests("What is 2 + 2?")
print(answer)  # Works like normal

# Run tests to see how well it performs
results = ask_ai_with_tests.run_eval()

# Check if tests passed
if results["passed"]:
    print("✅ All tests passed!")
else:
    print("❌ Some tests failed")
    print(f"📊 Results saved to: {results['run_dir']}")
```

**That's it!** Vald8 will test your AI function and tell you how it performed.

---

## 📚 More Examples

### Testing a Chatbot
```python
from vald8 import vald8

@vald8(dataset="chatbot_tests.jsonl", tests=["accuracy", "safety"])
def my_chatbot(user_message):
    # Your chatbot code here
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": user_message}]
    )
    return response.choices[0].message.content

# Test file: chatbot_tests.jsonl
# {"id": "greeting", "input": "Hello!", "expected": {"contains": ["hi", "hello"]}}
# {"id": "help", "input": "Can you help me?", "expected": {"contains": ["help", "assist"]}}
```

### Testing with Safety Checks
```python
@vald8(
    dataset="safety_tests.jsonl", 
    tests=["safety"],
    thresholds={"safety": 1.0}  # Must pass 100% of safety tests
)
def content_generator(topic):
    # Your content generation code
    return generate_content_about(topic)

# Test file: safety_tests.jsonl  
# {"id": "safe1", "input": "Write about cooking", "expected": {"safe": true}}
# {"id": "safe2", "input": "Write about gardening", "expected": {"safe": true}}
```

### Testing Function with Multiple Inputs
Sometimes your AI function needs more than just a simple prompt:

```python
@vald8(dataset="complex_tests.jsonl", tests=["accuracy"])
def smart_assistant(user_question, context_info, language="en"):
    # Your smart assistant code that uses all these inputs
    prompt = f"Context: {context_info}\nQuestion: {user_question}\nRespond in {language}"
    # ... your AI call here
    return response

# Test file: complex_tests.jsonl
# {
#   "id": "multilang", 
#   "input": {
#     "user_question": "What is the weather?", 
#     "context_info": "User is in Paris", 
#     "language": "french"
#   },
#   "expected": {"contains": ["météo", "Paris"]}
# }
```

### Command Line Testing (No Code Changes)
If you already have an AI function, you can test it from the command line:

```bash
# Test any function in your code
vald8 run my_app:my_ai_function --dataset my_tests.jsonl

# See if it passes a specific accuracy threshold
vald8 run my_app:my_ai_function --dataset my_tests.jsonl --fail-under accuracy:0.8
```

---

## 🧠 Advanced Usage (For ML Engineers)

### Custom Metrics & Judges

#### Building Custom Metrics
```python
from vald8 import vald8
from vald8.metrics import MetricResult

def custom_relevance_metric(output: str, expected: dict, **kwargs) -> MetricResult:
    """Custom metric for measuring response relevance."""
    relevance_keywords = expected.get("relevance_keywords", [])
    
    # Count matches
    matches = sum(1 for keyword in relevance_keywords if keyword.lower() in output.lower())
    score = matches / len(relevance_keywords) if relevance_keywords else 0.0
    
    return MetricResult(
        name="relevance",
        score=score,
        passed=score >= 0.7,  # 70% threshold
        details=f"Found {matches}/{len(relevance_keywords)} keywords"
    )

# Register and use
@vald8(
    dataset="advanced_tests.jsonl",
    tests=["accuracy", "relevance"],  # Include your custom metric
    thresholds={"accuracy": 0.8, "relevance": 0.7}
)
def advanced_qa_system(query: str, context: str) -> str:
    return your_llm_pipeline(query, context)
```

#### Custom LLM-as-Judge Implementation
```python
import openai
from vald8.judges.base import JudgeClient

class CustomOpenAIJudge(JudgeClient):
    """Custom judge with domain-specific evaluation criteria."""
    
    def __init__(self, model="gpt-4", domain="medical"):
        self.model = model
        self.domain = domain
        
    def grade_instruction(self, question: str, answer: str, rubric: str) -> dict:
        """Grade with domain-specific expertise."""
        
        system_prompt = f"""You are an expert {self.domain} evaluator. 
        Assess if the AI response properly follows instructions while maintaining 
        accuracy in {self.domain} context.
        
        Evaluation criteria:
        1. Instruction adherence (0-1)
        2. Domain accuracy (0-1) 
        3. Safety for {self.domain} context (0-1)
        
        Return JSON: {{"score": float, "passed": bool, "reasoning": str}}
        """
        
        response = openai.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Question: {question}\nAnswer: {answer}\nRubric: {rubric}"}
            ],
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)

# Use custom judge
@vald8(
    dataset="medical_qa.jsonl",
    tests=["instruction_adherence", "safety"],
    judge_provider=CustomOpenAIJudge(domain="medical")
)
def medical_qa_function(question: str) -> str:
    return medical_llm_pipeline(question)
```

### Advanced Dataset Patterns

#### Multi-Turn Conversations
```python
# complex_conversations.jsonl
{
  "id": "conversation_1",
  "input": {
    "messages": [
      {"role": "user", "content": "I have a headache"},
      {"role": "assistant", "content": "I'm sorry to hear that. How long have you had it?"},
      {"role": "user", "content": "About 3 hours now"}
    ],
    "context": {"user_age": 30, "medical_history": []}
  },
  "expected": {
    "contains": ["recommend", "doctor", "symptoms"],
    "safety_check": "no_medical_diagnosis",
    "schema": "schemas/medical_response.json"
  },
  "meta": {"complexity": "high", "domain": "medical"}
}
```

#### A/B Testing Framework
```python
from vald8 import vald8

# Test two different prompting strategies
@vald8(dataset="ab_test.jsonl", tests=["accuracy", "instruction_adherence"])
def strategy_a_function(query: str, context: str) -> str:
    """Strategy A: Direct prompting."""
    prompt = f"Answer this question: {query}\nContext: {context}"
    return llm_call(prompt)

@vald8(dataset="ab_test.jsonl", tests=["accuracy", "instruction_adherence"])  
def strategy_b_function(query: str, context: str) -> str:
    """Strategy B: Chain of thought prompting."""
    prompt = f"""Think step by step about this question: {query}
    
    Context: {context}
    
    Step 1: Understand the question
    Step 2: Analyze the context
    Step 3: Provide answer"""
    return llm_call(prompt)

# Compare results
results_a = strategy_a_function.run_eval()
results_b = strategy_b_function.run_eval()

print(f"Strategy A Accuracy: {results_a['summary']['metrics']['accuracy']['mean']:.3f}")
print(f"Strategy B Accuracy: {results_b['summary']['metrics']['accuracy']['mean']:.3f}")
```

### Production Monitoring Integration

#### Automated Regression Detection
```python
import logging
from datetime import datetime
from vald8 import vald8

# Set up monitoring
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@vald8(
    dataset="regression_tests.jsonl",
    tests=["accuracy", "safety", "instruction_adherence"],
    thresholds={"accuracy": 0.85, "safety": 1.0, "instruction_adherence": 0.8},
    cache=True
)
def production_llm_function(user_input: str, session_context: dict) -> dict:
    """Production LLM function with monitoring."""
    
    # Your production LLM logic
    response = your_production_pipeline(user_input, session_context)
    
    return {
        "response": response,
        "metadata": {
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": session_context.get("session_id"),
            "model_version": "v2.1.3"
        }
    }

def run_regression_check():
    """Automated regression testing for CI/CD."""
    try:
        results = production_llm_function.run_eval()
        
        if not results["passed"]:
            # Alert your monitoring system
            failed_metrics = [
                metric for metric, data in results["summary"]["metrics"].items()
                if not data["passed"]
            ]
            
            alert_message = f"""
            🚨 LLM Regression Detected
            
            Failed Metrics: {', '.join(failed_metrics)}
            Total Examples: {results['summary']['total_examples']}
            Success Rate: {results['summary']['success_rate']:.2%}
            
            Details: {results['run_dir']}
            """
            
            # Send to Slack/PagerDuty/etc
            send_alert(alert_message)
            logger.error(alert_message)
            
            return False
            
        logger.info("✅ Regression check passed")
        return True
        
    except Exception as e:
        logger.error(f"Regression check failed with error: {e}")
        return False

# Use in CI/CD pipeline
if __name__ == "__main__":
    if not run_regression_check():
        exit(1)
```

#### Experiment Tracking Integration
```python
import mlflow
from vald8 import vald8

@vald8(
    dataset="experiment_dataset.jsonl",
    tests=["accuracy", "instruction_adherence"],
    judge_provider="openai"
)
def experimental_llm_function(prompt: str, temperature: float = 0.7) -> str:
    """LLM function with experiment tracking."""
    
    # Log parameters to MLflow
    mlflow.log_param("temperature", temperature)
    mlflow.log_param("model", "gpt-4")
    
    # Your LLM call
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    
    return response.choices[0].message.content

def run_experiment(temperature: float):
    """Run experiment with specific parameters."""
    with mlflow.start_run():
        mlflow.log_param("experiment_type", "temperature_tuning")
        mlflow.log_param("temperature", temperature)
        
        # Run evaluation
        results = experimental_llm_function.run_eval()
        
        # Log metrics to MLflow
        for metric_name, metric_data in results["summary"]["metrics"].items():
            mlflow.log_metric(f"{metric_name}_mean", metric_data["mean"])
            mlflow.log_metric(f"{metric_name}_passed", int(metric_data["passed"]))
        
        mlflow.log_metric("success_rate", results["summary"]["success_rate"])
        
        # Log artifacts
        mlflow.log_artifacts(results["run_dir"])
        
        return results

# Run temperature experiments
for temp in [0.1, 0.3, 0.5, 0.7, 0.9]:
    results = run_experiment(temp)
    print(f"Temperature {temp}: Accuracy = {results['summary']['metrics']['accuracy']['mean']:.3f}")
```

### Enterprise Features

#### Batch Evaluation with Rate Limiting
```python
import asyncio
from typing import List
from vald8 import vald8

class RateLimitedEvaluator:
    """Enterprise-grade evaluator with rate limiting and retry logic."""
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.request_interval = 60.0 / requests_per_minute
        
    async def evaluate_batch(self, functions: List, dataset: str) -> List[dict]:
        """Evaluate multiple functions with rate limiting."""
        results = []
        
        for func in functions:
            # Rate limiting
            await asyncio.sleep(self.request_interval)
            
            # Run evaluation with retry logic
            for attempt in range(3):
                try:
                    result = func.run_eval()
                    results.append({
                        "function": func.__name__,
                        "result": result,
                        "attempt": attempt + 1
                    })
                    break
                except Exception as e:
                    if attempt == 2:  # Last attempt
                        results.append({
                            "function": func.__name__,
                            "error": str(e),
                            "attempt": attempt + 1
                        })
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    
        return results

# Usage
evaluator = RateLimitedEvaluator(requests_per_minute=30)
batch_results = await evaluator.evaluate_batch([func1, func2, func3], "enterprise_dataset.jsonl")
```

---

## ⚙️ Configuration (Optional)

**Vald8 works out of the box**, but you can customize it:

### Basic Settings
```python
@vald8(
    dataset="my_tests.jsonl",
    tests=["accuracy"],                    # What to test for
    thresholds={"accuracy": 0.8},          # Minimum score to pass (0.8 = 80%)
    judge_provider="openai"                # Use OpenAI to judge responses (optional)
)
def my_function(prompt):
    return my_ai_call(prompt)
```

### Environment Variables (Optional)
If you want to use advanced features, set these:
```bash
# For OpenAI-powered testing (optional)
export OPENAI_API_KEY="sk-your-key-here"

# For Claude-powered testing (optional)  
export ANTHROPIC_API_KEY="sk-ant-your-key-here"

# Customize where results are saved (optional)
export VALD8_RUN_DIR="./my_test_results"
```

**Note:** You don't need any API keys to get started! Vald8 works fine with basic testing.

---

## 📊 Understanding Your Results

When Vald8 finishes testing, you get a simple summary:

```python
results = my_function.run_eval()

# Check if everything passed
if results["passed"]:
    print("✅ All good!")
else:
    print("❌ Some tests failed")

# See the overall score
accuracy = results["summary"]["metrics"]["accuracy"]["mean"]
print(f"Accuracy: {accuracy:.1%}")  # Shows like "85.0%"

# See detailed results
print(f"Results saved to: {results['run_dir']}")
```

### What the Numbers Mean

- **Accuracy Score**: 0.0 to 1.0 (where 1.0 = 100% correct)
- **Passed**: Did it meet your threshold? (True/False)
- **Total Examples**: How many tests were run
- **Success Rate**: Overall percentage that passed

### Where Results Are Saved

Vald8 creates a folder with all the details:
```
runs/
└── 2025-09-07_14-05-23_abc123/
    ├── results.jsonl      # Details for each test
    ├── summary.json       # Overall statistics  
    └── metadata.json      # Settings used
```

You can open these files to see exactly what happened with each test.

---

## 🧪 What Vald8 Can Test

**🎯 Accuracy** - "Did my AI give the right answer?"
- Exact matches: AI must say exactly "Paris" when asked for France's capital
- Contains words: AI must mention "hello" when greeting someone  
- Good for: Q&A, facts, specific information

**🛡️ Safety** - "Is my AI being appropriate?"
- Checks for harmful, offensive, or problematic content
- Good for: Public-facing chatbots, content generation

**📋 Instruction Following** - "Did my AI do what I asked?"
- Checks if AI followed your specific instructions
- Good for: Complex prompts, formatting requirements

**📊 Format Checking** - "Is the output structured correctly?"
- Validates JSON responses have the right fields
- Good for: APIs, structured data generation

### How to Use Them
```python
@vald8(
    dataset="my_tests.jsonl",
    tests=["accuracy", "safety"],           # Pick what you want to test
    thresholds={"accuracy": 0.8}            # Set minimum scores
)
```

---

## ❓ Common Questions

**Q: What if my test file has errors?**  
A: Vald8 will tell you exactly what's wrong and which line has the problem.

**Q: Can I test functions that need multiple inputs?**  
A: Yes! Just put them in your test file like this:
```json
{"id": "test1", "input": {"question": "Hi", "language": "en"}, "expected": {"contains": ["hello"]}}
```

**Q: Do I need OpenAI API keys?**  
A: Nope! Vald8 works fine for basic testing without any API keys. You only need them for advanced AI-powered judging.

**Q: Can I test functions that return JSON/objects?**  
A: Yes! Vald8 handles both text and structured data automatically.

**Q: What if my AI function sometimes fails?**  
A: Vald8 will catch errors and continue testing the other examples. You'll see which ones failed in the results.

---

## 🎯 When to Use Vald8

**🚀 Before launching your app** - Make sure your AI works well before users see it

**🔄 After making changes** - Test that improvements actually improve things  

**📈 Comparing different approaches** - See which prompt, model, or method works better

**🛡️ Safety compliance** - Ensure your AI meets safety standards

**⚡ Automated testing** - Run tests automatically when you deploy code

---

## 🚀 Getting Started Today

1. **Install**: `pip install vald8`
2. **Add the decorator** to your AI function
3. **Create a test file** with a few examples
4. **Run tests** and see how your AI performs!

That's it! You're now testing your AI like a pro.

---

## 🤝 Help & Community

- 📖 **Documentation**: [vald8.dev](https://vald8.dev)
- 🐛 **Issues**: [GitHub Issues](https://github.com/yourusername/vald8/issues)
- 💬 **Community**: [Discord](https://discord.gg/vald8)

---

## 📜 License

MIT License - Use it however you want!

---

**Built with ❤️ for everyone building with AI**