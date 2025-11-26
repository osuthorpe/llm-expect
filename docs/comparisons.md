# Comparisons

LLM Expect is opinionated. It's not the right tool for everyone. Here is how it compares to other popular LLM evaluation tools.

## vs. DeepEval / Ragas
**Best for:** RAG pipelines, detailed metrics, academic benchmarks.

| Feature | LLM Expect | DeepEval / Ragas |
| :--- | :--- | :--- |
| **Philosophy** | Integration Testing | Metric Research |
| **Complexity** | Low | High |
| **Setup** | 1 Decorator | SDK + Configuration |
| **Metrics** | Practical (Accuracy, Schema) | Academic (Faithfulness, Relevancy) |

**Choose LLM Expect if:** You want to ensure your function doesn't break in CI.
**Choose DeepEval/Ragas if:** You are researching the optimal RAG retrieval strategy.

## vs. Promptfoo
**Best for:** Comparing prompts across many models via CLI.

| Feature | LLM Expect | Promptfoo |
| :--- | :--- | :--- |
| **Language** | Python Native | Node.js / YAML |
| **Interface** | Decorator | CLI / Web View |
| **Logic** | Python Functions | Static Prompts |

**Choose LLM Expect if:** Your LLM logic is complex Python code (tools, chains).
**Choose Promptfoo if:** You are A/B testing raw prompts across 10 different models.

## vs. LangSmith / Arize
**Best for:** Production observability and tracing.

| Feature | LLM Expect | LangSmith |
| :--- | :--- | :--- |
| **Stage** | Pre-deployment (Testing) | Post-deployment (Monitoring) |
| **Data** | Local | Cloud |
| **Cost** | Free | Paid |

**Choose LLM Expect if:** You want a local test runner.
**Choose LangSmith if:** You need to see what your users are sending to your app in production.
