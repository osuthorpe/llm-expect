# ⚙️ Configuration Reference

## Decorator Parameters

Configure LLM Expect using the `@llm_expect` decorator:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `dataset` | `str` | **Required** | Path to JSONL file (relative or absolute). |
| `tests` | `list[str]` | `[]` | Metrics to evaluate: `["accuracy", "schema_fidelity", "safety", "custom_judge"]`. |
| `thresholds` | `dict` | `{"accuracy": 0.8}` | Pass/fail thresholds per metric. |
| `judge_provider` | `str` | `None` | LLM judge provider: `"openai"`, `"anthropic"`, `"bedrock"`. |
| `judge_model` | `str` | Provider default | Specific model for the judge (e.g., `"gpt-4"`). |
| `sample_size` | `int` | `None` (All) | Number of examples to sample from the dataset. |
| `shuffle` | `bool` | `False` | Whether to shuffle examples before sampling. |
| `cache` | `bool` | `True` | Enable caching of results to avoid re-running passed tests. |
| `cache_dir` | `str` | `".vald8_cache"` | Directory for cache files. |
| `results_dir` | `str` | `"runs"` | Directory to save detailed evaluation results. |
| `parallel` | `bool` | `False` | Run tests in parallel (faster for IO-bound) |
| `fail_fast` | `bool` | `False` | Stop evaluation immediately on the first failure. |
| `timeout` | `int` | `60` | Timeout in seconds for the decorated function execution. |

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
@llm_expect(dataset="tests.jsonl")  # Other config from env vars
def generate(prompt: str) -> str:
    # Your function
    pass
```
