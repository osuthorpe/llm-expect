# Product Philosophy: The Vald8 Way

Vald8 is not just a library; it's an opinionated approach to AI engineering. Our product philosophy is built on three pillars: **Minimalism**, **Developer Experience (DX)**, and **Transparency**.

## 1. Minimalism: Less is More

In a world of bloated frameworks and complex orchestration tools, Vald8 chooses to be small.

*   **No New Infrastructure**: We don't ask you to spin up a Docker container, a database, or a separate evaluation server. Vald8 runs where your code runs.
*   **No "Platform" Lock-in**: We are not a SaaS platform trying to capture your data. We are a library that empowers your existing workflow.
*   **Single Responsibility**: Vald8 does one thing well: it validates LLM outputs against expectations. It doesn't try to be a prompt playground, a vector database, or a fine-tuning framework.

> "Perfection is achieved, not when there is nothing more to add, but when there is nothing left to take away." - Antoine de Saint-Exupéry

## 2. Developer Experience (DX) First

We believe that if a tool is hard to use, it won't be used. Evaluation is critical, so it must be effortless.

*   **Zero-Config Start**: You should be able to run your first evaluation in under 5 minutes.
*   **Intuitive APIs**: We use Python decorators (`@vald8`) because they are idiomatic and non-intrusive. You don't rewrite your functions; you just annotate them.
*   **Standard Formats**: We use JSONL for datasets. It's human-readable, diff-able, and streams efficiently. No proprietary binary formats or complex SQL schemas.

## 3. Transparency: No Black Boxes

Trust is the currency of AI. If you don't understand why a model failed, you can't fix it.

*   **Open Metrics**: Our "judges" are just prompts. We document exactly what we ask the LLM judge. You can inspect, modify, and override them.
*   **Traceable Results**: Every evaluation run produces a clear, structured JSONL log. You can see exactly what the input was, what the model output, and why the judge passed or failed it.
*   **Code over Magic**: We prefer explicit code over implicit "magic". If something happens, you should be able to find the line of code that caused it.


## 4. Built on Human-Centered AI Principles

Vald8 is the practical implementation of the [10 Principles of Human-Centered AI Design](https://www.alexthorpe.com/principles). It provides the tooling necessary to turn these abstract principles into concrete engineering practices.

*   **Principle 2 (Reduce Surprise) & 4 (Constrain Systems)**: Vald8's core purpose is to define boundaries. By setting explicit expectations (e.g., `contains`, `regex`, `safety`), you ensure your system behaves predictably and stays within defined limits.
*   **Principle 5 (Evaluate Continuously)**: Behavior shifts over time. Vald8 enables "evaluation as code," making continuous regression testing a natural part of your daily workflow, not just a pre-launch checklist.
*   **Principle 7 (Make Failures Observable)**: Vald8's detailed JSONL logs make failures explicit, discoverable, and debuggable. We treat bias and refusals as system failures that must be surfaced, never silenced.
*   **Principle 10 (Document Decisions)**: A Vald8 dataset is more than just a test file; it is documentation. It records exactly what the system is expected to do, serving as a living record of your product decisions.

## Who is Vald8 For?

*   **The Pragmatic Engineer**: Who wants to ship reliable AI features without drowning in complexity.
*   **The Product Leader**: Who needs to know "is this actually better?" before approving a release.
*   **The AI Researcher**: Who needs a lightweight, reproducible way to benchmark experiments.

## Who is Vald8 NOT For?

*   Teams looking for a heavy, all-in-one enterprise AI platform (though Vald8 can integrate with them).
*   Users who prefer GUI-based "no-code" evaluation tools over code-centric workflows.
