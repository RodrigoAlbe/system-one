"""
Data structures and primitive definitions for System One.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union


@dataclass
class Choice:
    """
    Selects one option from a defined set with calibrated confidence.
    """
    options: List[str]
    instructions: str
    type: str = "choice"


@dataclass
class Noul:
    """
    Yes/No probabilistic question. Returns probability float between 0.0 and 1.0.
    """
    instructions: str
    type: str = "noul"


@dataclass
class Score:
    """
    Evaluates degree along an ordered scale (e.g. ['Low', 'Medium', 'High'] or ['P1', 'P2', 'P3']).
    """
    levels: List[str]
    instructions: str
    type: str = "score"


@dataclass
class QuestionResult:
    """The result of a single question evaluation."""
    question_id: str
    question_type: str
    value: Union[str, float, int]
    confidence: float
    raw_distribution: Optional[Dict[str, float]] = None


@dataclass
class EvaluationMetrics:
    """Telemetry and cost metrics for the evaluation."""
    latency_ms: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    provider: str
    model: str


@dataclass
class EvaluationResponse:
    """Consolidated response holding answers for all parallel questions and performance metrics."""
    answers: Dict[str, QuestionResult]
    metrics: EvaluationMetrics
    raw_response: Dict[str, Any]
