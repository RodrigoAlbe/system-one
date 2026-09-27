"""Structured AI decisions with local response validation."""

from .primitives import (
    Choice,
    Noul,
    Score,
    QuestionResult,
    EvaluationMetrics,
    EvaluationResponse,
)
from .client import SystemOneClient
from .errors import (
    InvalidResponseError,
    ProviderRefusalError,
    IncompleteResponseError,
    ProviderError,
    EvaluationTimeoutError,
)
from .logprobs import softmax, entropy_confidence

__version__ = "2.0.0"
__all__ = [
    "SystemOneClient",
    "InvalidResponseError",
    "ProviderRefusalError",
    "IncompleteResponseError",
    "ProviderError",
    "EvaluationTimeoutError",
    "Choice",
    "Noul",
    "Score",
    "QuestionResult",
    "EvaluationMetrics",
    "EvaluationResponse",
    "softmax",
    "entropy_confidence",
]
