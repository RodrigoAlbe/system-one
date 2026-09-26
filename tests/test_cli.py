import json
from unittest.mock import Mock

import pytest

from system_one import (
    EvaluationMetrics,
    EvaluationResponse,
    QuestionResult,
    InvalidResponseError,
    ProviderError,
)
from system_one.cli import main


@pytest.fixture
def factory(monkeypatch):
    factory = Mock()
    factory.return_value.evaluate.return_value = EvaluationResponse(
        {"q": QuestionResult("q", "choice", "Billing", 0.8)},
        EvaluationMetrics(1, None, 2, None, None, "ollama", "test-model"),
        {"private": "raw provider details"},
    )
    monkeypatch.setattr("system_one.cli.SystemOneClient", factory)
    return factory


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["unknown", "state"],
        ["choice", "state"],
        ["choice", "state", "instructions"],
        ["score", "state", "instructions"],
        ["noul", "state", "instructions", "unexpected"],
        ["choice", "state", "instructions", "A", "A"],
        ["score", "state", "instructions", ""],
        ["noul", "state", ""],
        ["noul", "state", "--provider", "unknown"],
    ],
)
def test_invalid_arguments_fail_before_client_creation(factory, capsys, argv):
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2
    factory.assert_not_called()
    output = capsys.readouterr()
    assert output.out == ""
    assert "error:" in output.err


@pytest.mark.parametrize(
    "argv",
    [
        [
            "--provider",
            "ollama",
            "--model",
            "custom",
            "--json",
            "choice",
            "state",
            "Route",
            "Billing",
            "Support",
        ],
        [
            "choice",
            "state",
            "Route",
            "Billing",
            "Support",
            "--provider",
            "ollama",
            "--model",
            "custom",
            "--json",
        ],
        [
            "choice",
            "state",
            "--provider",
            "ollama",
            "Route",
            "Billing",
            "--model",
            "custom",
            "Support",
            "--json",
        ],
    ],
)
def test_flags_and_json_output(factory, capsys, argv):
    assert main(argv) == 0
    factory.assert_called_once_with(provider="ollama", model="custom")
    state, questions = factory.return_value.evaluate.call_args.args
    assert state == "state"
    assert questions["q"].options == ["Billing", "Support"]
    assert questions["q"].instructions == "Route"
    output = capsys.readouterr()
    data = json.loads(output.out)
    assert data["answers"]["q"]["value"] == "Billing"
    assert data["metrics"]["input_tokens"] is None
    assert "raw_response" not in data
    assert "private" not in output.out
    assert output.err == ""


@pytest.mark.parametrize(
    "kind,values,value,label",
    [
        ("choice", ["Billing", "Support"], "Billing", "Selected: Billing"),
        ("score", ["Low", "High"], "High", "Score Level: High"),
        ("noul", [], 0.8, "Probability (True/Yes): 0.80"),
    ],
)
def test_human_output(factory, capsys, kind, values, value, label):
    factory.return_value.evaluate.return_value.answers["q"].value = value
    assert main([kind, "state", "instructions", *values]) == 0
    assert capsys.readouterr().out.strip() == label


def test_noul_keeps_optional_instructions(factory):
    factory.return_value.evaluate.return_value.answers["q"].value = 0.5
    assert main(["noul", "state"]) == 0
    assert (
        factory.return_value.evaluate.call_args.args[1]["q"].instructions
        == "Evaluate the provided state"
    )


@pytest.mark.parametrize(
    "error",
    [
        InvalidResponseError("Invalid answer"),
        ProviderError("Unavailable"),
        ValueError("Missing API key"),
    ],
)
@pytest.mark.parametrize("json_output", [False, True])
def test_runtime_errors_have_nonzero_exit_and_stderr(
    factory, capsys, error, json_output
):
    factory.return_value.evaluate.side_effect = error
    assert main(["noul", "state", *(["--json"] if json_output else [])]) == 1
    output = capsys.readouterr()
    assert output.out == ""
    if json_output:
        assert json.loads(output.err)["error"]["type"] == type(error).__name__
    else:
        assert str(error) in output.err
    assert "Traceback" not in output.err


def test_help_is_offline(factory, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    factory.assert_not_called()
    assert "--provider" in capsys.readouterr().out
