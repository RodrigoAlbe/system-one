"""
System One Native - Drop-in implementation of TypeSafe / Jev decision primitives.
Fast, structured, calibrated decisions with zero-cost free tier options.
"""

from .primitives import (
    Choice,
    Noul,
    Score,
    QuestionResult,
    EvaluationMetrics,
    EvaluationResponse,
)
from .client import SystemOneClient
from .logprobs import softmax, entropy_confidence

__version__ = "1.1.0"
__all__ = [
    "SystemOneClient",
    "Choice",
    "Noul",
    "Score",
    "QuestionResult",
    "EvaluationMetrics",
    "EvaluationResponse",
    "softmax",
    "entropy_confidence",
]

