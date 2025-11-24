# Judge Providers

Vald8 supports multiple LLM providers for judge-based evaluation.

## OpenAI

Use OpenAI models (GPT-4, GPT-3.5, etc.) as judges.

### Configuration

```python
from vald8 import vald8

@vald8(
    dataset="tests.jsonl",
    tests=["custom_judge"],
    judge_provider="openai",
    judge_model="gpt-4"
)
def generate(prompt: str) -> str:
    # Your function
    pass
```

### Environment Variables

```bash
export OPENAI_API_KEY=your-key-here
export VALD8_JUDGE_MODEL=gpt-4
```

### Supported Models

- `gpt-4`
- `gpt-4-turbo`
- `gpt-5.1`
- `gpt-3.5-turbo`

## Anthropic

Use Claude models as judges.

### Configuration

```python
@vald8(
    dataset="tests.jsonl",
    tests=["custom_judge"],
    judge_provider="anthropic",
    judge_model="claude-3-opus-20240229"
)
```

### Environment Variables

```bash
export ANTHROPIC_API_KEY=your-key-here
export VALD8_JUDGE_MODEL=claude-3-opus-20240229
```

### Supported Models

- `claude-3-opus-20240229`
- `claude-3-sonnet-20240229`
- `claude-3-haiku-20240307`

## AWS Bedrock

Use Bedrock-hosted models as judges.

### Configuration

```python
@vald8(
    dataset="tests.jsonl",
    tests=["custom_judge"],
    judge_provider="bedrock",
    judge_model="anthropic.claude-3-sonnet-20240229-v1:0"
)
```

### Environment Variables

```bash
export AWS_ACCESS_KEY_ID=your-key
export AWS_SECRET_ACCESS_KEY=your-secret
export AWS_REGION=us-east-1
export VALD8_JUDGE_MODEL=anthropic.claude-3-sonnet-20240229-v1:0
```

## Judge Configuration Options

### Model Selection

```python
@vald8(
    judge_model="gpt-4"  # Specify model
)
```

### API Key

```python
from vald8.models import JudgeConfig

judge_config = JudgeConfig(
    provider="openai",
    model="gpt-4",
    api_key="your-key"  # Or use environment variable
)
```

### Custom Base URL

```python
judge_config = JudgeConfig(
    provider="openai",
    model="gpt-4",
    base_url="https://custom-api.example.com/v1"
)
```

### Timeout and Retries

```python
judge_config = JudgeConfig(
    provider="openai",
    model="gpt-4",
    timeout=60,  # seconds
    max_retries=5
)
```

### Temperature

```python
judge_config = JudgeConfig(
    provider="openai",
    model="gpt-4",
    temperature=0.0  # 0.0 for deterministic
)
```

## Judge Evaluation Types

### Custom Judge

Evaluate with custom criteria:

```json
{
  "expected": {
    "judge": {
      "prompt": "Is the response polite and professional?"
    }
  }
}
```

### Instruction Adherence

Evaluate instruction following:

```python
@vald8(
    dataset="tests.jsonl",
    tests=["instruction_adherence"],
    judge_provider="openai"
)
```

### Safety Evaluation

Evaluate safety with LLM judge:

```python
@vald8(
    dataset="tests.jsonl",
    tests=["safety"],
    judge_provider="openai"
)
```

## Best Practices

1. **Use GPT-4 for complex evaluations**: More reliable for nuanced judgments
2. **Set temperature to 0.0**: For consistent, deterministic evaluations
3. **Cache results**: Enable caching to avoid redundant API calls
4. **Monitor costs**: Judge evaluations make additional API calls
5. **Test judge prompts**: Validate that judge criteria match your needs
