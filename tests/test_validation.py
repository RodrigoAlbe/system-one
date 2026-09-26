import json

import pytest

from system_one import (
    Choice,
    Noul,
    Score,
    SystemOneClient,
    InvalidResponseError,
    ProviderRefusalError,
    IncompleteResponseError,
)


def envelope(content):
    return {"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}


def parse(content, questions=None):
    client = SystemOneClient(provider="openai", api_key="test")
    return client._parse_response(
        envelope(content), questions or {"q": Noul("Urgent?")}, 1
    )


@pytest.mark.parametrize(
    "content",
    [
        "{}",
        "null",
        "[]",
        '"text"',
        "invalid",
        "```json\n{}\n```",
        "",
        '{"q":null}',
        '{"q":0.9}',
        '{"q":{"probability":0.5}}',
        '{"q":{"probability":0.5,"confidence":0.5,"extra":true}}',
        '{"q":{"probability":0.5,"confidence":0.5},"extra":{}}',
        '{"q":{"probability":0,"probability":1,"confidence":1}}',
        '{"q":{"probability":0,"confidence":1},"q":{"probability":1,"confidence":1}}',
    ],
)
def test_invalid_answers_are_not_silently_defaulted(content):
    with pytest.raises(InvalidResponseError):
        parse(content)


def test_oversized_numeric_literal_is_an_invalid_response():
    with pytest.raises(InvalidResponseError):
        parse('{"q":{"probability":' + "9" * 5000 + ',"confidence":1}}')


@pytest.mark.parametrize("field", ["probability", "confidence"])
@pytest.mark.parametrize(
    "value",
    [
        -0.1,
        1.7,
        True,
        False,
        "0.9",
        None,
        float("nan"),
        float("inf"),
        -float("inf"),
        10**400,
    ],
)
def test_numbers_are_strict_and_bounded(field, value):
    answer = {"probability": 0.5, "confidence": 0.5, field: value}
    with pytest.raises(InvalidResponseError):
        parse(json.dumps({"q": answer}))


@pytest.mark.parametrize("value", [0, 0.5, 1])
def test_valid_boundary_values(value):
    result = parse(json.dumps({"q": {"probability": value, "confidence": value}}))
    assert result.answers["q"].value == value
    assert result.answers["q"].confidence_source == "model_reported"
    assert result.metrics.estimated_cost_usd is None


@pytest.mark.parametrize(
    "question,field",
    [(Choice(["A", "B"], "Pick"), "selected"), (Score(["A", "B"], "Rank"), "level")],
)
@pytest.mark.parametrize("value", ["C", "a", 1, True, None, ["A"]])
def test_categorical_values_must_match_options(question, field, value):
    with pytest.raises(InvalidResponseError):
        parse(json.dumps({"q": {field: value, "confidence": 1}}), {"q": question})


@pytest.mark.parametrize("use_logprobs", [False, True])
def test_logprobs_cannot_overwrite_batch_answers(use_logprobs):
    questions = {
        "first": Choice(["A", "B"], "First"),
        "second": Choice(["A", "B"], "Second"),
        "flag": Noul("Yes?"),
        "level": Score(["A", "B"], "Level"),
    }
    body = {
        "first": {"selected": "A", "confidence": 0.8},
        "second": {"selected": "B", "confidence": 0.7},
        "flag": {"probability": 0.3, "confidence": 0.6},
        "level": {"level": "B", "confidence": 0.5},
    }
    data = envelope(json.dumps(body))
    data["choices"][0]["logprobs"] = {
        "content": [
            {
                "token": "A",
                "top_logprobs": [
                    {"token": "A", "logprob": -0.01},
                    {"token": "B", "logprob": -4},
                ],
            },
            {
                "token": "B",
                "top_logprobs": [
                    {"token": "A", "logprob": -3},
                    {"token": "B", "logprob": -0.1},
                ],
            },
            {
                "token": "1",
                "top_logprobs": [
                    {"token": "1", "logprob": -0.01},
                    {"token": "0", "logprob": -8},
                ],
            },
        ]
    }
    result = SystemOneClient(
        provider="openai", api_key="test", use_logprobs=use_logprobs
    )._parse_response(data, questions, 1)
    assert [a.value for a in result.answers.values()] == ["A", "B", 0.3, "B"]
    assert [a.confidence for a in result.answers.values()] == [0.8, 0.7, 0.6, 0.5]
    assert all(a.raw_distribution is None for a in result.answers.values())
    assert result.raw_response == data


@pytest.mark.parametrize(
    "data",
    [
        None,
        [],
        {},
        {"choices": []},
        {"choices": [None]},
        {"choices": [{"message": None}]},
        envelope(None),
    ],
)
def test_malformed_envelopes(data):
    with pytest.raises(InvalidResponseError):
        SystemOneClient(provider="openai", api_key="test")._parse_response(
            data, {"q": Noul("Q")}, 1
        )


@pytest.mark.parametrize(
    "reason,error",
    [
        ("length", IncompleteResponseError),
        ("content_filter", ProviderRefusalError),
        ("tool_calls", IncompleteResponseError),
    ],
)
def test_finish_reason_checked_even_for_valid_json(reason, error):
    data = envelope('{"q":{"probability":1,"confidence":1}}')
    data["choices"][0]["finish_reason"] = reason
    with pytest.raises(error):
        SystemOneClient(provider="openai", api_key="test")._parse_response(
            data, {"q": Noul("Q")}, 1
        )


def test_refusal_has_distinct_error():
    data = envelope(None)
    data["choices"][0]["message"]["refusal"] = "Refused"
    with pytest.raises(ProviderRefusalError):
        SystemOneClient(provider="openai", api_key="test")._parse_response(
            data, {"q": Noul("Q")}, 1
        )


def test_gemini_joins_answer_parts_and_ignores_thoughts():
    data = {
        "candidates": [
            {
                "finishReason": "STOP",
                "content": {
                    "parts": [
                        {"text": "private reasoning", "thought": True},
                        {"text": '{"q":{"probability":'},
                        {"text": '1,"confidence":0.9}}'},
                    ]
                },
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 5,
            "totalTokenCount": 15,
        },
    }
    result = SystemOneClient(provider="gemini", api_key="test")._parse_response(
        data, {"q": Noul("Q")}, 1
    )
    assert result.answers["q"].value == 1
    assert result.metrics.total_tokens == 15


@pytest.mark.parametrize(
    "data,error",
    [
        ({"promptFeedback": {"blockReason": "SAFETY"}}, ProviderRefusalError),
        ({"candidates": [{"finishReason": "SAFETY"}]}, ProviderRefusalError),
        ({"candidates": [{"finishReason": "MAX_TOKENS"}]}, IncompleteResponseError),
        ({"candidates": []}, InvalidResponseError),
    ],
)
def test_gemini_failed_generations(data, error):
    with pytest.raises(error):
        SystemOneClient(provider="gemini", api_key="test")._parse_response(
            data, {"q": Noul("Q")}, 1
        )
