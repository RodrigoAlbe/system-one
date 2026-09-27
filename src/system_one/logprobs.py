"""Token-level diagnostics, not calibrated decision probabilities."""

from __future__ import annotations
import math
from typing import Dict, List, Optional


def softmax(logprobs: Dict[str, float]) -> Dict[str, float]:
    """
    Computes numerically stable softmax over candidate log probabilities:
    P(i) = exp(logp_i - max_logp) / sum(exp(logp_j - max_logp))
    """
    if not logprobs:
        return {}

    if any(not math.isfinite(value) for value in logprobs.values()):
        raise ValueError("logprobs must be finite")
    max_logp = max(logprobs.values())
    exp_vals = {k: math.exp(v - max_logp) for k, v in logprobs.items()}
    sum_exp = sum(exp_vals.values())

    if sum_exp == 0:
        uniform = 1.0 / len(logprobs)
        return {k: uniform for k in logprobs}

    return {k: v / sum_exp for k, v in exp_vals.items()}


def entropy_confidence(probs: Dict[str, float]) -> float:
    """
    Computes distribution concentration, not empirical correctness probability:
    Confidence = 1.0 - (Entropy / Max_Entropy)

    Returns 1.0 if probability is concentrated in a single option,
    and 0.0 if distribution is completely uniform (maximum uncertainty).
    """
    if not probs or any(
        type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1
        for p in probs.values()
    ):
        raise ValueError("probabilities must be a non-empty finite distribution")
    if not math.isclose(math.fsum(probs.values()), 1.0, rel_tol=1e-6, abs_tol=1e-9):
        raise ValueError("probabilities must sum to 1")
    n = len(probs)
    if n <= 1:
        return 1.0

    entropy = 0.0
    for p in probs.values():
        if p > 1e-9:
            entropy -= p * math.log(p)

    max_entropy = math.log(n)
    if max_entropy == 0:
        return 1.0

    normalized_entropy = entropy / max_entropy
    confidence = max(0.0, min(1.0, 1.0 - normalized_entropy))
    return round(confidence, 4)


def _extract_position(steps, position, target_tokens, probability_field):
    if not steps:
        return {}
    if position is None:
        if len(steps) != 1:
            raise ValueError(
                "Specify a token position; probabilities across positions cannot be pooled"
            )
        position = 0
    if type(position) is not int or not 0 <= position < len(steps):
        raise ValueError("Token position is out of range")
    result = {}
    for item in steps[position]:
        token = item.get("token", "")
        # Retain exact token text. Whitespace/quotes/case are part of token identity.
        if target_tokens is None or token in target_tokens:
            value = item[probability_field]
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("Token log probability must be finite")
            result[token] = float(value)
    return result


def extract_openai_logprobs(
    choice_data: dict,
    target_tokens: Optional[List[str]] = None,
    *,
    position: Optional[int] = None,
) -> Dict[str, float]:
    """Read alternatives at one token position; never combine different contexts."""
    content = (choice_data.get("logprobs") or {}).get("content") or []
    steps = [item.get("top_logprobs") or [] for item in content]
    return _extract_position(steps, position, target_tokens, "logprob")


def extract_gemini_logprobs(
    candidate_data: dict,
    target_tokens: Optional[List[str]] = None,
    *,
    position: Optional[int] = None,
) -> Dict[str, float]:
    """Read Gemini topCandidates[position].candidates as token diagnostics."""
    result = candidate_data.get("logprobsResult") or {}
    steps = [item.get("candidates") or [] for item in result.get("topCandidates", [])]
    return _extract_position(steps, position, target_tokens, "logProbability")
