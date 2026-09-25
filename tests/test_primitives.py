import pytest
from system_one.primitives import Choice, Noul, Score, QuestionResult, EvaluationMetrics, EvaluationResponse


def test_choice_creation():
    c = Choice(instructions="Pick department", options=["Sales", "Billing"])
    assert c.type == "choice"
    assert len(c.options) == 2
    assert "Sales" in c.options


def test_noul_creation():
    n = Noul(instructions="Is this urgent?")
    assert n.type == "noul"
    assert n.instructions == "Is this urgent?"


def test_score_creation():
    s = Score(instructions="Assess risk", levels=["Low", "Medium", "High"])
    assert s.type == "score"
    assert len(s.levels) == 3
