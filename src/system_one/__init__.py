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
)
from .logprobs import softmax, entropy_confidence

__version__ = "1.1.0"
__all__ = [
    "SystemOneClient",
    "InvalidResponseError",
    "ProviderRefusalError",
    "IncompleteResponseError",
    "ProviderError",
    "Choice",
    "Noul",
    "Score",
    "QuestionResult",
    "EvaluationMetrics",
    "EvaluationResponse",
    "softmax",
    "entropy_confidence",
]
