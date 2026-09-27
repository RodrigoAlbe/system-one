# Independent synthetic validation

30 authored cases, unrelated to Machina: 16 objective, 6 ambiguous/invalid and
8 adversarial. Choice, Score and Noul have explicit rules and labels fixed before
live calls. The deterministic baseline gets 30/30; it is the reference rule
implementation, not independent human validation or evidence that AI adds value.
The model should match these rules. Perfect accuracy here does not establish
real-world accuracy, safety, calibration or resistance to arbitrary injection.
Do not tune prompts against these cases and continue calling them held out.

From a repository checkout, with the provider key already configured:

```bash
python -m pip install -e .
python scripts/validate_independent.py
python scripts/validate_independent.py --provider gemini --output independent-report.json
```

Replace gemini with groq or openai to use the corresponding environment variable.
Use --model to select a model explicitly. Do not commit or paste credentials.
The offline command makes no API calls. The live command attempts 30 evaluations,
each with up to 3 HTTP attempts and a 35-second evaluation budget. Provider usage
may be billed; no second model or repetitions are run automatically.

The JSON contains individual answers, accuracy, group counts, failures, review
counts, successful latency p50/p95, token coverage, model/configuration, dataset
hashes and Git commit. Exit 1 indicates a wrong answer or failed request; setup
errors also terminate unsuccessfully. Investigate each error instead of masking
it with aggregate accuracy. Compare independent reports for another provider only
when needed. Never mistake an offline baseline result for model performance.

No live result is included: the preparation environment has no supported API key.
