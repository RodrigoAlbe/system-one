import json
from types import SimpleNamespace

import pytest

from benchmark import load_cases, run_benchmark
from system_one import (
    Choice,
    Noul,
    EvaluationMetrics,
    EvaluationResponse,
    QuestionResult,
    InvalidResponseError,
    ProviderError,
)


def client(outcomes):
    pending = iter(outcomes)

    def evaluate(*args):
        item = next(pending)
        if isinstance(item, Exception):
            raise item
        return item

    return SimpleNamespace(
        evaluate=evaluate,
        provider="mock",
        model="mock",
        timeout=35,
        max_retries=3,
        use_logprobs=False,
        capabilities=SimpleNamespace(response_mode="json_schema"),
    )


def result(latency, value=0.8):
    return EvaluationResponse(
        {
            "flag": QuestionResult("flag", "noul", value, 0.9),
            "route": QuestionResult("route", "choice", "A", 0.9),
        },
        EvaluationMetrics(latency, 10, 5, 15, None, "mock", "mock"),
        {},
    )


def cases():
    return [
        {
            "id": "test",
            "state": "s",
            "questions": {"flag": Noul("Q"), "route": Choice(["A", "B"], "Q")},
            "expected": {"flag": True, "route": "A"},
        }
    ]


def test_quality_and_latency_report():
    report = run_benchmark(client([result(100), result(300, 0.2)]), cases(), repeats=2)
    assert report["successful_latency_p50_ms"] == 200
    assert report["successful_latency_p95_ms"] == 300
    assert report["accuracy_on_valid_answers"] == 0.75
    assert report["noul_brier_score"] == pytest.approx(0.34)
    assert report["successful_total_tokens"] == 30
    assert report["estimated_cost_usd"] is None
    assert report["unique_cases"] == 1
    assert len(report["dataset_sha256"]) == 64
    json.dumps(report, allow_nan=False)


def test_failures_remain_visible_in_denominator():
    report = run_benchmark(
        client([result(100), InvalidResponseError("bad"), ProviderError("down")]),
        cases(),
        repeats=3,
    )
    assert report["accuracy_on_valid_answers"] == 1
    assert report["correct_over_attempted_labels"] == pytest.approx(1 / 3)
    assert report["invalid_response_rate"] == pytest.approx(1 / 3)
    assert report["provider_failures"] == 1


def test_all_failed_run_has_no_fabricated_metrics():
    report = run_benchmark(client([ProviderError("down")]), cases())
    assert report["successful_latency_p50_ms"] is None
    assert report["noul_brier_score"] is None
    assert report["accuracy_on_valid_answers"] is None
    assert report["correct_over_attempted_labels"] == 0
    assert report["successful_total_tokens"] is None
    assert report["reported_total_tokens"] is None
    assert report["token_usage_coverage"]["total_tokens"]["fraction"] is None


def test_unlabeled_smoke_cases_do_not_claim_accuracy():
    case = cases()[0]
    del case["expected"]
    report = run_benchmark(client([result(1)]), [case])
    assert report["accuracy_on_valid_answers"] is None
    assert report["correct_over_attempted_labels"] is None


def test_dataset_roundtrip_and_label_validation(tmp_path):
    path = tmp_path / "cases.json"
    raw = [
        {
            "id": "test",
            "state": "s",
            "questions": {"flag": {"type": "noul", "instructions": "Q"}},
            "expected": {"flag": True},
        }
    ]
    path.write_text(json.dumps(raw), encoding="utf-8")
    assert isinstance(load_cases(path)[0]["questions"]["flag"], Noul)
    raw[0]["expected"]["flag"] = 1
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="boolean"):
        load_cases(path)


@pytest.mark.parametrize(
    "data", [[], {}, [{}], [{"id": "x", "state": "s", "questions": {}, "expected": {}}]]
)
def test_invalid_datasets_rejected(tmp_path, data):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        load_cases(path)


def test_partial_usage_does_not_undercount_as_a_complete_total():
    first, second = result(1), result(2)
    second.metrics.input_tokens = None
    second.metrics.total_tokens = None
    report = run_benchmark(client([first, second]), cases(), repeats=2)
    assert report["successful_input_tokens"] is None
    assert report["successful_total_tokens"] is None
    assert report["reported_input_tokens"] == 10
    assert report["reported_total_tokens"] == 15
    assert report["successful_output_tokens"] == 10
    assert report["token_usage_coverage"]["input_tokens"] == {
        "reported_responses": 1,
        "missing_responses": 1,
        "fraction": 0.5,
    }
    assert report["token_usage_coverage"]["output_tokens"]["fraction"] == 1
    assert report["records"][1]["metrics"]["input_tokens"] is None


@pytest.mark.parametrize("value", [None, 0])
def test_unknown_usage_and_measured_zero_are_not_conflated(value):
    response = result(1)
    response.metrics.input_tokens = value
    response.metrics.output_tokens = value
    response.metrics.total_tokens = value
    report = run_benchmark(client([response]), cases())
    assert report["successful_total_tokens"] == value
    assert report["reported_total_tokens"] == value
    assert report["token_usage_coverage"]["total_tokens"]["fraction"] == (
        0 if value is None else 1
    )
    json.dumps(report, allow_nan=False)
