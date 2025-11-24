# Roadmap

This roadmap outlines the strategic direction for Vald8. It is a living document and will evolve based on community feedback and the changing landscape of AI.

## Short Term (v0.2.0 - v0.3.0)
*Focus: Deepening the core validation capabilities.*

*   **Expanded Judge Providers**: Support for Google Gemini, Azure OpenAI, and local models (via Ollama/vLLM).
*   **More Built-in Metrics**:
    *   `ToneMetric`: Validate the sentiment/tone of responses.
    *   `JsonSchemaMetric`: Strict validation of JSON outputs against Pydantic models.
    *   `HallucinationMetric`: RAG-specific checks for faithfulness to context.
*   **Parallel Execution**: Speed up evaluations by running judge calls in parallel.

## Medium Term (v0.5.0 - v1.0.0)
*Focus: Integration and Workflow.*

*   **CI/CD Integration**: First-class GitHub Actions and GitLab CI runners to block PRs on regression.
*   **Dashboarding**: A lightweight, local CLI dashboard (`vald8 ui`) to visualize results over time without needing a hosted server.
*   **Dataset Generation**: Tools to help users bootstrap evaluation datasets from their existing logs ("Golden Set" creation).
*   **Cost Tracking**: Estimate and track the cost of evaluation runs.

## Long Term (v1.0+)
*Focus: Community and Ecosystem.*

*   **Plugin System**: Allow the community to write and share custom judges and metrics as plugins.
*   **Hosted Hub (Optional)**: A place to share public benchmarks and datasets.
*   **Language Support**: TypeScript/JS SDK for Node.js environments.

## Strategic Themes

1.  **"Evaluation as Code"**: We will continue to double down on code-centric workflows.
2.  **Model Agnostic**: Vald8 will never favor one model provider over another.
3.  **Community Driven**: We prioritize features requested by open-source contributors.

---
*Have a feature request? [Open an issue](https://github.com/osuthorpe/vald8-sdk/issues) or contribute to the discussion!*
