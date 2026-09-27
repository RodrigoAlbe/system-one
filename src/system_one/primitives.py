"""
Data structures and primitive definitions for System One.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union


def _validate_instructions(instructions: str) -> None:
    if not isinstance(instructions, str) or not instructions.strip():
        raise ValueError("instructions must be a non-empty string")


def _validate_options(options: List[str]) -> None:
    if not isinstance(options, list) or not options:
        raise ValueError("options/levels must be a non-empty list of strings")
    if any(not isinstance(option, str) or not option.strip() for option in options):
        raise ValueError("options/levels must contain non-empty strings")
    if len(set(options)) != len(options):
        raise ValueError("options/levels must be unique")


@dataclass
class Choice:
    """
    Selects one option with model-reported, uncalibrated confidence.
    """

    options: List[str]
    instructions: str
    type: str = "choice"

    def __post_init__(self):
        _validate_instructions(self.instructions)
        _validate_options(self.options)
        if self.type != "choice":
            raise ValueError("Choice.type must be 'choice'")


@dataclass
class Noul:
    """
    Yes/No probabilistic question. Returns probability float between 0.0 and 1.0.
    """

    instructions: str
    type: str = "noul"

    def __post_init__(self):
        _validate_instructions(self.instructions)
        if self.type != "noul":
            raise ValueError("Noul.type must be 'noul'")


@dataclass
class Score:
    """
    Evaluates degree along an ordered scale (e.g. ['Low', 'Medium', 'High'] or ['P1', 'P2', 'P3']).
    """

    levels: List[str]
    instructions: str
    type: str = "score"

    def __post_init__(self):
        _validate_instructions(self.instructions)
        _validate_options(self.levels)
        if self.type != "score":
            raise ValueError("Score.type must be 'score'")


@dataclass
class QuestionResult:
    """The result of a single question evaluation."""

    question_id: str
    question_type: str
    value: Union[str, float, int]
    confidence: float
    raw_distribution: Optional[Dict[str, float]] = None
    confidence_source: str = "model_reported"


@dataclass
class EvaluationMetrics:
    """Telemetry and cost metrics for the evaluation."""

    latency_ms: float
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    total_tokens: Optional[int]
    estimated_cost_usd: Optional[float]
    provider: str
    model: str


@dataclass
class EvaluationResponse:
    """Consolidated response holding answers for all parallel questions and performance metrics."""

    answers: Dict[str, QuestionResult]
    metrics: EvaluationMetrics
    raw_response: Dict[str, Any]
