import pytest
from system_one.primitives import Choice, Noul, Score


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


@pytest.mark.parametrize("options", [[], "A", [""], [1], ["A", "A"], [" "]])
@pytest.mark.parametrize("primitive", [Choice, Score])
def test_invalid_options(primitive, options):
    with pytest.raises(ValueError):
        primitive(options, "Question")


@pytest.mark.parametrize("instructions", [None, "", "  ", 1])
def test_invalid_instructions(instructions):
    with pytest.raises(ValueError):
        Noul(instructions)


def test_mutated_question_revalidated_before_request():
    from system_one import SystemOneClient

    q = Choice(["A", "B"], "Pick")
    q.options.clear()
    with pytest.raises(ValueError):
        SystemOneClient(provider="openai", api_key="test")._prepare_request(
            "s", {"q": q}
        )


@pytest.mark.parametrize(
    "questions", [{}, [], {"": Noul("Q")}, {1: Noul("Q")}, {"q": object()}]
)
def test_invalid_question_map(questions):
    from system_one import SystemOneClient

    with pytest.raises(ValueError):
        SystemOneClient(provider="openai", api_key="test")._prepare_request(
            "s", questions
        )
