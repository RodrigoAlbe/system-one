# System One Native

Structured AI decisions with multiple providers and local response validation.

Use `Choice`, `Noul`, and `Score` to classify application state without parsing chat
prose. Evaluate several questions in one request with Gemini, Groq, OpenAI, or a
local Ollama server. Decisions remain probabilistic: valid JSON does not guarantee
that a classification is correct.

## Installation

Install from GitHub (requires Git and Python 3.9+):

```bash
python -m pip install "git+https://github.com/RodrigoAlbe/system-one.git"
```

There is currently no public PyPI release. The GitHub command installs `main`;
append `@<commit-sha>` to the URL to pin a particular revision.

For development, install from a checkout:

```bash
git clone https://github.com/RodrigoAlbe/system-one.git
cd system-one
python -m pip install -e .
```

Agent skill:

```bash
npx skills add RodrigoAlbe/system-one --global
```

Claude Code plugin:

```bash
claude plugin marketplace add RodrigoAlbe/system-one
claude plugin install system-one@system-one
```

The skill/plugin installs agent instructions. Install the Python library separately
when you want to execute those examples. Maintainers can validate both manifests:

```bash
claude plugin validate --strict .claude-plugin/marketplace.json
claude plugin validate --strict .claude-plugin/plugin.json
```

## Quickstart

Set `GEMINI_API_KEY`, or select a different provider explicitly.

```python
from system_one import SystemOneClient, Choice, Noul, Score

client = SystemOneClient(provider="gemini")  # Close after all evaluations.
response = client.evaluate(
    state={"message": "Payment gateway returning 500 errors", "plan": "Enterprise"},
    questions={
        "department": Choice(options=["Backend", "Billing", "DevOps"], instructions="Routing department"),
        "urgent": Noul(instructions="Is there an active revenue-impacting outage?"),
        "severity": Score(levels=["P1", "P2", "P3", "P4"], instructions="Operational severity"),
    },
)
print(response.answers["department"].value)
print(response.answers["urgent"].value)  # Model-estimated number in [0, 1]
print(response.metrics.output_tokens)  # None if the provider omitted usage
print(response.metrics.estimated_cost_usd)  # None: unknown, not zero
```

Keyword arguments avoid confusion: positional constructors are
`Choice(options, instructions)` and `Score(levels, instructions)`.

Convenience methods and async evaluation use the same validation:

```python
category = client.choice("I love it", "Sentiment", ["Positive", "Neutral", "Negative"])
probability = client.noul("Unsolicited sales email", "Is this spam?")
priority = client.score("Disk almost full", "Severity", ["Low", "Medium", "High"])
client.close()

# Inside an async function:
# async with SystemOneClient(provider="gemini") as async_client:
#     response = await async_client.evaluate_async("log payload", {"alert": Noul("Page on-call?")})
```

## Validation and failure handling

Every answer must contain exactly the expected fields. Missing/extra question
IDs, duplicate JSON keys, out-of-range values, booleans masquerading as numbers,
unknown options, non-finite numbers, and malformed JSON raise
`InvalidResponseError`. A batch is atomic: no partial result is returned.
Question options/levels must be non-empty lists of unique, non-empty strings.

```python
from system_one import InvalidResponseError, ProviderError

try:
    result = client.evaluate("message", {"urgent": Noul("Is this urgent?")})
except InvalidResponseError:
    print("No usable decision: send to manual review")
except ProviderError:
    print("Provider unavailable: queue for a later attempt")
else:
    print(result.answers["urgent"].value)
```

`ProviderRefusalError` and `IncompleteResponseError` are subclasses of
`InvalidResponseError`. Refused or truncated responses never become default
negative answers. Invalid responses are not automatically retried.

HTTP 408/429/500/502/503/504 and transport failures are retried with backoff,
respecting `Retry-After`. `max_retries=3` retains its historical meaning of **three
total attempts**. `timeout` is an HTTP operation timeout, not a total deadline;
`total_timeout=35.0` is the evaluation budget shared by all attempts and waits.
A retry wait that cannot fit fails immediately with `EvaluationTimeoutError`
(a `ProviderError`) so the caller can reschedule without violating `Retry-After`.
Async HTTP I/O is cancelled at the remaining deadline. **Sync deadlines are
cooperative**: each HTTP phase is limited to the remaining budget and overdue
results are rejected, but an in-flight synchronous request cannot be interrupted
at an exact wall-clock deadline. Use async evaluation when cancellation at a
deadline is required. Synchronous preparation/parsing and event-loop scheduling
also mean this is not a hard real-time guarantee.

Clients now keep connection pools. Use `with SystemOneClient(...) as client` for
sync calls and `async with SystemOneClient(...) as client` for async calls, or
call `close()` / `await aclose()` explicitly. Reuse async clients within one event
loop; close them before that loop exits. A closed client cannot be reused.
If both APIs were used, `await aclose()` closes both pools. Do not close a client
while evaluations are in flight. Convenience methods use these same pools.

`ProviderError` exposes `provider`, `status_code`, `retryable`, `retry_after`
(seconds or `None`), `attempts`, and `request_id` (when available). Error messages
omit provider response bodies. `retryable` describes the failure, not a promise
that another attempt will succeed; authentication errors are not retryable.

```python
from system_one import SystemOneClient, Noul, EvaluationTimeoutError

with SystemOneClient(provider="gemini", total_timeout=10) as client:
    try:
        result = client.evaluate("service unavailable", {"outage": Noul("Is there an outage?")})
    except EvaluationTimeoutError as exc:
        print("Queue for later", exc.retry_after, exc.attempts)
```


## Providers and response formats

| Provider | Environment variable | Default model | Automatic response mode |
| --- | --- | --- | --- |
| Gemini | `GEMINI_API_KEY` | `gemini-3.1-flash-lite` | JSON Schema via `responseJsonSchema` |
| Groq | `GROQ_API_KEY` | `llama-3.3-70b-versatile` | JSON object + schema in prompt |
| OpenAI | `OPENAI_API_KEY` | `gpt-4o-mini` | Strict JSON Schema |
| Ollama | None | `qwen2.5:7b` | JSON Schema on the local OpenAI-compatible endpoint |

Auto-detection checks Gemini, Groq, then OpenAI keys. With no key it selects Gemini
and reports the missing key before making a request. For local inference use
`SystemOneClient(provider="ollama")` explicitly.

Groq `openai/gpt-oss-20b` and `openai/gpt-oss-120b` use strict schema mode. The known
OpenAI schema models are `gpt-4o-mini`, `gpt-4o-mini-2024-07-18`, and
`gpt-4o-2024-08-06`. Other OpenAI/Groq model names conservatively use JSON object
mode. Both modes include the complete schema in the prompt and validate locally.

For a custom model with verified support, set `response_mode="json_schema"`.
Use `response_mode="json_object"` for older compatible servers. Overrides do not
make an unsupported model support a feature; provider errors remain explicit.

Provider contracts: [Gemini API](https://ai.google.dev/api/generate-content#v1beta.GenerationConfig),
[Groq structured outputs](https://console.groq.com/docs/structured-outputs),
[OpenAI structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs),
[Ollama structured outputs](https://docs.ollama.com/capabilities/structured-outputs).
Availability, quotas, latency, and billing depend on the provider, model, and account.
This project does not guarantee free usage or sub-second latency.

## Confidence and diagnostic logprobs

`confidence` and `Noul.value` are **model-reported estimates**, not calibrated
probabilities of correctness. `confidence_source` is `"model_reported"`.
`raw_distribution` remains `None`; truncated top-token alternatives cannot establish
an option-level distribution, especially across multiple questions or tokens.

`use_logprobs=False` is the default. Opt-in requests are supported for OpenAI and
Ollama endpoints that implement them; data is kept only in `raw_response`.
Groq and Gemini diagnostic opt-in currently raises `ValueError` locally.
Token logprobs never overwrite the JSON answer or its reported confidence.

The diagnostic helpers `extract_openai_logprobs` and `extract_gemini_logprobs`
require `position=` for multi-token responses. They preserve exact token text and
never pool probabilities across positions. `entropy_confidence` measures
concentration, not accuracy. Calibrate decision thresholds on independent labeled
data before using them for automated actions.

## Tests and measured benchmarks

```bash
pip install -e ".[dev]"
python -m pytest

# Live calls; requires provider credentials and may incur usage charges:
python tests/benchmark.py --provider gemini --repeat 5 --output benchmark.json
python tests/benchmark.py --provider ollama --output local-benchmark.json
python tests/benchmark.py --provider groq --dataset cases.json --output labeled-benchmark.json
```

The three bundled scenarios are unlabeled smoke examples, not an accuracy study.
The report includes model/configuration, timestamp, package/Python versions,
dataset hash, per-request results, validation/provider failures, token usage, and
successful-request latency p50/p95 (nearest-rank p95). Latency includes retries.
Token totals cover successful responses only; failed requests may consume tokens.
Cost remains unknown. No competitor or savings figures are fabricated.

Token fields are `None` (JSON `null`) when missing or null in a provider response;
an explicit zero remains zero. Each `successful_*_tokens` total is only populated
when every successful response reported that field. Otherwise it is null.
`reported_*_tokens` sums the available measurements and is null when none exist.
`token_usage_coverage` reports measured/missing response counts and the fraction
covered per field, using successful responses as the denominator. With no successful
responses, totals and coverage fractions are null.

A labeled dataset is a JSON array. Each case needs a unique `id`, `state`,
`questions`, and an `expected` label for every question:

```json
[
  {
    "id": "example-only",
    "state": "The server is unavailable to all users.",
    "questions": {
      "outage": {"type": "noul", "instructions": "Is the server unavailable?"}
    },
    "expected": {"outage": true}
  }
]
```

Use `options` for choice questions and `levels` for score questions; their labels
must be one of those strings. Noul labels are booleans. Reports add accuracy on
valid answers, correct answers over all attempted labels (including failures), and
Noul Brier score. Noul accuracy uses a 0.5 threshold. Repetitions do not increase
the number of independent cases. A representative, independently reviewed dataset
is still needed before claiming real-world quality or calibrated confidence.

## Migration to 2.0.0

Version 2.0.0 includes the validation changes below and these additional changes:

- Evaluations default to a 35-second total budget; configure `total_timeout` for
  longer tasks. Long retry waits fail promptly instead of sleeping unboundedly.
- Clients own persistent connection pools and must be closed explicitly.
- `entropy_confidence` rejects empty, non-finite, negative and unnormalized
  distributions; `softmax` preserves full precision instead of rounding each value.
- `ProviderError` carries structured diagnostics; `EvaluationTimeoutError` is new.

### Changes from the original 1.1.0 behavior

- Invalid or missing answers now raise errors instead of becoming default values.
- Logprobs are disabled by default and no longer rewrite values or confidence.
- Unsupported diagnostic logprob requests fail before network access.
- `estimated_cost_usd` is now optional and returns `None` when unknown.
- `input_tokens`, `output_tokens`, and `total_tokens` are optional too; missing
  usage is no longer recorded as zero. Benchmark totals include coverage metadata.
- Empty/duplicate options and invalid question definitions are rejected locally.
- Multi-position logprob extraction requires an explicit token position.

## CLI

```bash
system-one noul "Customer requests a refund" "Is the customer requesting a refund?"
system-one choice "Server CPU at 99%" "Action" Scale Restart Ignore
system-one score "Database disk at 92%" "Severity" Low Medium High Critical
system-one --provider ollama --model qwen2.5:7b --json choice "Payment failed" "Department" Billing Support
```

`choice` and `score` require instructions followed by at least one explicit
option/level. They never supply placeholder options. `noul` accepts optional
instructions and rejects extra options. Run `system-one --help` for usage.
`--provider`, `--model`, and `--json` can appear before or after positional arguments.
Credentials come from the provider environment variables described above.

With `--json`, stdout contains `answers` (the single question ID is `q`) and
`metrics`; it excludes raw provider data. Successful runs exit 0. Invalid command
arguments exit 2 with argparse diagnostics on stderr. Evaluation/configuration
failures exit 1 with a short error on stderr (a JSON `error` object when `--json`
is set), leaving stdout empty.

## Decision-quality evaluation

Start with [the evaluation protocol](docs/evaluation.md) and the synthetic,
explicitly labeled [triage fixture](examples/triage.synthetic.json). These cases
exercise the workflow; they are not customer data, independent evidence, or a
production quality claim. The protocol defines a manual-review outcome and the
real held-out dataset required before enabling automated actions.

## License

MIT.
