import pytest
from system_one.logprobs import (
    softmax,
    entropy_confidence,
    extract_openai_logprobs,
    extract_gemini_logprobs,
)


def test_softmax_calculation():
    # log(1.0) = 0.0, log(0.5) = -0.693
    raw_logprobs = {"A": -0.1, "B": -2.3, "C": -4.5}
    probs = softmax(raw_logprobs)
    assert abs(sum(probs.values()) - 1.0) < 0.01
    assert probs["A"] > probs["B"] > probs["C"]


def test_entropy_confidence_certain():
    # 100% confidence in one choice
    probs = {"A": 1.0, "B": 0.0}
    conf = entropy_confidence(probs)
    assert conf == 1.0


def test_entropy_confidence_uncertain():
    # 50/50 split -> maximum entropy -> confidence ~0.0
    probs = {"A": 0.5, "B": 0.5}
    conf = entropy_confidence(probs)
    assert conf == 0.0


def test_extract_openai_logprobs():
    mock_choice = {
        "logprobs": {
            "content": [
                {
                    "token": "Backend",
                    "logprob": -0.15,
                    "top_logprobs": [
                        {"token": "Backend", "logprob": -0.15},
                        {"token": "Billing", "logprob": -2.10},
                        {"token": "Support", "logprob": -3.50},
                    ],
                }
            ]
        }
    }
    extracted = extract_openai_logprobs(
        mock_choice, target_tokens=["Backend", "Billing", "Support"]
    )
    assert "Backend" in extracted
    assert "Billing" in extracted
    assert extracted["Backend"] == -0.15

    probs = softmax(extracted)
    assert probs["Backend"] > 0.8
    conf = entropy_confidence(probs)
    assert conf > 0.5  # Significant confidence


def test_positions_cannot_be_pooled():
    data = {
        "logprobs": {
            "content": [
                {"top_logprobs": [{"token": "A", "logprob": -0.1}]},
                {"top_logprobs": [{"token": "B", "logprob": -0.2}]},
            ]
        }
    }
    with pytest.raises(ValueError, match="position"):
        extract_openai_logprobs(data)
    assert extract_openai_logprobs(data, position=1) == {"B": -0.2}


def test_gemini_top_candidates_shape():
    data = {
        "logprobsResult": {
            "topCandidates": [
                {"candidates": [{"token": "A", "logProbability": -0.1}]},
                {"candidates": [{"token": "B", "logProbability": -0.2}]},
            ],
            "chosenCandidates": [{"token": "A", "logProbability": -0.1}],
        }
    }
    assert extract_gemini_logprobs(data, position=1) == {"B": -0.2}
    with pytest.raises(ValueError):
        extract_gemini_logprobs(data)


def test_token_identity_is_preserved():
    data = {
        "logprobs": {
            "content": [
                {
                    "top_logprobs": [
                        {"token": " A", "logprob": -0.1},
                        {"token": "A", "logprob": -0.2},
                    ]
                }
            ]
        }
    }
    assert extract_openai_logprobs(data, ["A"]) == {"A": -0.2}


@pytest.mark.parametrize("value", [float("inf"), float("nan"), -float("inf")])
def test_nonfinite_softmax_inputs_rejected(value):
    with pytest.raises(ValueError):
        softmax({"A": value})
