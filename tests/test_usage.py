import pytest

from system_one import SystemOneClient, Noul, InvalidResponseError


def parse_usage(provider, usage, omitted=False):
    content = '{"q":{"probability":0.5,"confidence":0.5}}'
    if provider == "gemini":
        data = {
            "candidates": [
                {"finishReason": "STOP", "content": {"parts": [{"text": content}]}}
            ]
        }
        key = "usageMetadata"
    else:
        data = {"choices": [{"finish_reason": "stop", "message": {"content": content}}]}
        key = "usage"
    if not omitted:
        data[key] = usage
    return (
        SystemOneClient(provider=provider, api_key="test")
        ._parse_response(data, {"q": Noul("Q")}, 1)
        .metrics
    )


@pytest.mark.parametrize("provider", ["gemini", "openai", "groq", "ollama"])
@pytest.mark.parametrize("usage,omitted", [(None, True), (None, False), ({}, False)])
def test_unknown_usage_stays_unknown(provider, usage, omitted):
    metrics = parse_usage(provider, usage, omitted)
    assert (metrics.input_tokens, metrics.output_tokens, metrics.total_tokens) == (
        None,
        None,
        None,
    )


@pytest.mark.parametrize(
    "provider,fields",
    [
        ("gemini", ("promptTokenCount", "candidatesTokenCount", "totalTokenCount")),
        ("openai", ("prompt_tokens", "completion_tokens", "total_tokens")),
    ],
)
def test_partial_and_zero_usage_are_distinct(provider, fields):
    metrics = parse_usage(provider, {fields[0]: 0, fields[1]: None})
    assert (metrics.input_tokens, metrics.output_tokens, metrics.total_tokens) == (
        0,
        None,
        None,
    )
    metrics = parse_usage(provider, dict(zip(fields, [0, 0, 0])))
    assert (metrics.input_tokens, metrics.output_tokens, metrics.total_tokens) == (
        0,
        0,
        0,
    )
    metrics = parse_usage(provider, {fields[0]: 10, fields[1]: 5})
    assert (
        metrics.total_tokens is None
    )  # Never infer totals that may include reasoning tokens.


@pytest.mark.parametrize(
    "provider,field", [("gemini", "promptTokenCount"), ("openai", "prompt_tokens")]
)
@pytest.mark.parametrize("value", [True, -1, 1.5, "10", [], float("nan")])
def test_invalid_reported_counts_rejected(provider, field, value):
    with pytest.raises(InvalidResponseError):
        parse_usage(provider, {field: value})


@pytest.mark.parametrize("usage", [[], False, 0, ""])
def test_invalid_usage_container_is_not_treated_as_missing(usage):
    with pytest.raises(InvalidResponseError):
        parse_usage("openai", usage)
