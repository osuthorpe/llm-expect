# Configuration

## Decorator Parameters

Configure Vald8 using the `@vald8` decorator:

```python
@vald8(
    dataset="path/to/dataset.jsonl",       # Required: Path to JSONL dataset
    tests=["accuracy", "schema_fidelity"], # Optional: Metrics to evaluate
    thresholds={"accuracy": 0.9},          # Optional: Pass/fail thresholds
    judge_provider="openai",               # Optional: LLM judge provider
    judge_model="gpt-4",                   # Optional: Judge model name
    sample_size=10,                        # Optional: Number of examples
    shuffle=True,                          # Optional: Shuffle before sampling
    cache=True,                            # Optional: Cache results
    cache_dir=".vald8_cache",              # Optional: Cache directory
    results_dir="runs",                    # Optional: Results directory
    fail_fast=False,                       # Optional: Stop on first failure
    timeout=60,                            # Optional: Function timeout (seconds)
)
```

## Environment Variables

All parameters can be set via environment variables with the `VALD8_` prefix:

| Variable | Type | Description | Default |
|----------|------|-------------|---------|
| `VALD8_TESTS` | List | Comma-separated metrics | `[]` |
| `VALD8_THRESHOLD` | Float | Global threshold | `0.8` |
| `VALD8_THRESHOLD_ACCURACY` | Float | Accuracy threshold | `0.8` |
| `VALD8_THRESHOLD_SAFETY` | Float | Safety threshold | `1.0` |
| `VALD8_SAMPLE_SIZE` | Int | Number of examples | All |
| `VALD8_SHUFFLE` | Bool | Shuffle examples | `false` |
| `VALD8_CACHE` | Bool | Enable caching | `true` |
| `VALD8_CACHE_DIR` | String | Cache directory | `.vald8_cache` |
| `VALD8_RESULTS_DIR` | String | Results directory | `runs` |
| `VALD8_FAIL_FAST` | Bool | Stop on first failure | `false` |
| `VALD8_TIMEOUT` | Int | Function timeout (seconds) | `60` |

## Judge Configuration

For LLM-as-judge metrics:

| Variable | Description | Default |
|----------|-------------|---------|
| `VALD8_JUDGE_MODEL` | Judge model name | Provider-specific |
| `VALD8_JUDGE_API_KEY` | Judge API key | From provider env var |
| `VALD8_JUDGE_BASE_URL` | Custom API base URL | Provider default |
| `VALD8_JUDGE_TIMEOUT` | Judge request timeout | `30` |
| `VALD8_JUDGE_MAX_RETRIES` | Max retry attempts | `3` |
| `VALD8_JUDGE_TEMPERATURE` | Judge temperature | `0.0` |

**Provider API Keys:**
- OpenAI: `OPENAI_API_KEY`
- Anthropic: `ANTHROPIC_API_KEY`
- Bedrock: `AWS_ACCESS_KEY_ID`

## Example: Using Environment Variables

```bash
export VALD8_TESTS="accuracy,safety"
export VALD8_THRESHOLD_ACCURACY=0.95
export VALD8_JUDGE_PROVIDER=openai
export OPENAI_API_KEY=your-key-here
```

```python
@vald8(dataset="tests.jsonl")  # Other config from env vars
def generate(prompt: str) -> str:
    # Your function
    pass
```
