# Changelog

## 2.0.0 — unreleased

Breaking changes:
- Default evaluation budget is 35 seconds across attempts/waits. Async requests
  can be cancelled at the deadline; synchronous deadlines are cooperative.
- Clients retain connections and require close/aclose or context managers.
- Invalid entropy distributions now raise ValueError. Softmax no longer rounds.
- Consolidates the response-validation, optional usage, and diagnostic-logprob
  behavior changes previously documented against the original 1.1.0.

Added:
- Structured ProviderError metadata and EvaluationTimeoutError.
- Tests for deadline handling, pool lifecycle, invalid probabilities and errors.
- Synthetic labeled triage fixture and held-out quality evaluation protocol.

No production accuracy, latency improvement or cost claim has been measured.
