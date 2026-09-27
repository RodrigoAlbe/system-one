"""Small synthetic evaluation; no live API calls unless --provider is supplied."""

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import subprocess

ROOT = Path(__file__).resolve().parents[1]
benchmark = runpy.run_path(str(ROOT / "tests/benchmark.py"))


def baseline(state, question):
    if question.type == "score":
        return (
            "Low" if state["load"] < 50 else "Medium" if state["load"] < 80 else "High"
        )
    if question.type == "noul":
        return state["enabled"] is True and state["verified"] is True
    status, paid = state.get("status"), state.get("paid")
    if (
        not isinstance(status, str)
        or status not in ("pending", "shipped", "cancelled")
        or type(paid) is not bool
    ):
        return "Review"
    if status == "shipped" and not paid:
        return "Review"
    return {
        "cancelled": "Cancelled",
        "shipped": "Shipped",
        "pending": "Ready" if paid else "AwaitingPayment",
    }[status]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=["gemini", "groq", "openai", "ollama"])
    parser.add_argument("--model")
    parser.add_argument("--output", type=Path, default=Path("independent-report.json"))
    args = parser.parse_args()
    dataset = ROOT / "examples/independent.synthetic.json"
    cases = benchmark["load_cases"](dataset)
    correct = sum(
        baseline(c["state"], c["questions"]["decision"]) == c["expected"]["decision"]
        for c in cases
    )
    baseline_result = {
        "correct": correct,
        "cases": len(cases),
        "accuracy": correct / len(cases),
    }
    if correct != len(cases):
        raise SystemExit(
            "Fixture labels disagree with deterministic rules; stop before API calls"
        )
    if not args.provider:
        print(
            json.dumps(
                {"mode": "offline", "baseline": baseline_result, "live_calls": 0}
            )
        )
        return 0
    from system_one import SystemOneClient

    with SystemOneClient(provider=args.provider, model=args.model) as client:
        report = benchmark["run_benchmark"](client, cases, repeats=1)
    report["baseline"] = baseline_result
    report["scope"] = (
        "Authored synthetic contract checks, not real-world quality evidence"
    )
    report["dataset_file_sha256"] = hashlib.sha256(dataset.read_bytes()).hexdigest()
    report["git_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    by_id = {c["id"]: c for c in cases}
    groups = {}
    for record in report["records"]:
        case = by_id[record["case_id"]]
        group = groups.setdefault(
            case["group"], {"attempted": 0, "correct": 0, "failed": 0, "review": 0}
        )
        group["attempted"] += 1
        if record["status"] != "ok":
            group["failed"] += 1
            continue
        value = record["answers"]["decision"]["value"]
        group["review"] += value == "Review"
        predicted = (
            value >= 0.5 if case["questions"]["decision"].type == "noul" else value
        )
        group["correct"] += predicted == case["expected"]["decision"]
    report["groups"] = groups
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "groups": groups,
                "accuracy": report["correct_over_attempted_labels"],
            }
        )
    )
    return int(
        bool(report["invalid_responses"] or report["provider_failures"])
        or report["correct_over_attempted_labels"] != 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
