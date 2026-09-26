"""
Mathematical calibration and logprobs extraction for System One decisions.
Computes true probability distributions and entropy-based confidence from token logits.
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple, Any


def softmax(logprobs: Dict[str, float]) -> Dict[str, float]:
    """
    Computes numerically stable softmax over candidate log probabilities:
    P(i) = exp(logp_i - max_logp) / sum(exp(logp_j - max_logp))
    """
    if not logprobs:
        return {}

    max_logp = max(logprobs.values())
    exp_vals = {k: math.exp(v - max_logp) for k, v in logprobs.items()}
    sum_exp = sum(exp_vals.values())

    if sum_exp == 0:
        uniform = 1.0 / len(logprobs)
        return {k: uniform for k in logprobs}

    return {k: round(v / sum_exp, 4) for k, v in exp_vals.items()}


def entropy_confidence(probs: Dict[str, float]) -> float:
    """
    Computes calibrated confidence based on normalized Shannon entropy:
    Confidence = 1.0 - (Entropy / Max_Entropy)
    
    Returns 1.0 if probability is concentrated in a single option,
    and 0.0 if distribution is completely uniform (maximum uncertainty).
    """
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


def extract_openai_logprobs(choice_data: dict, target_tokens: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Extracts token log probabilities from OpenAI / Groq / Ollama / vLLM response format.
    Looks inside choice_data['logprobs']['content'].
    """
    logprobs_info = choice_data.get("logprobs", {})
    if not logprobs_info:
        return {}

    content_tokens = logprobs_info.get("content", [])
    raw_logprobs: Dict[str, float] = {}

    for item in content_tokens:
        top_items = item.get("top_logprobs", [])
        for top in top_items:
            tok = top.get("token", "").strip().strip('"').strip("'")
            lp = float(top.get("logprob", -999.0))
            if tok and (target_tokens is None or any(tok.lower() == t.lower() for t in target_tokens)):
                # If target specified, match case-insensitively
                match = tok
                if target_tokens:
                    for t in target_tokens:
                        if tok.lower() == t.lower():
                            match = t
                            break
                if match not in raw_logprobs or lp > raw_logprobs[match]:
                    raw_logprobs[match] = lp

    return raw_logprobs


def extract_gemini_logprobs(candidate_data: dict, target_tokens: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Extracts token log probabilities from Google Gemini API response format:
    candidate_data['logprobsResult']['topCandidates'] or ['chosenCandidates'].
    """
    logprobs_res = candidate_data.get("logprobsResult", {})
    if not logprobs_res:
        return {}

    raw_logprobs: Dict[str, float] = {}

    # Check chosenCandidates with topCandidates
    chosen = logprobs_res.get("chosenCandidates", [])
    for c in chosen:
        top_candidates = c.get("topCandidates", [])
        for top in top_candidates:
            tok = top.get("token", "").strip().strip('"').strip("'")
            lp = float(top.get("logProbability", -999.0))
            if tok and (target_tokens is None or any(tok.lower() == t.lower() for t in target_tokens)):
                match = tok
                if target_tokens:
                    for t in target_tokens:
                        if tok.lower() == t.lower():
                            match = t
                            break
                if match not in raw_logprobs or lp > raw_logprobs[match]:
                    raw_logprobs[match] = lp

    return raw_logprobs
