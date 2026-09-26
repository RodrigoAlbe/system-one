import pytest
import math
from system_one.logprobs import softmax, entropy_confidence, extract_openai_logprobs, extract_gemini_logprobs


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
                        {"token": "Support", "logprob": -3.50}
                    ]
                }
            ]
        }
    }
    extracted = extract_openai_logprobs(mock_choice, target_tokens=["Backend", "Billing", "Support"])
    assert "Backend" in extracted
    assert "Billing" in extracted
    assert extracted["Backend"] == -0.15

    probs = softmax(extracted)
    assert probs["Backend"] > 0.8
    conf = entropy_confidence(probs)
    assert conf > 0.5  # Significant confidence
