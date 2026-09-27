"""Measured smoke benchmark; optional labeled datasets support quality evaluation."""

from __future__ import annotations
import argparse
import hashlib
import json
import math
import platform
import statistics
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from system_one import (
    SystemOneClient,
    Choice,
    Noul,
    Score,
    InvalidResponseError,
    ProviderError,
    __version__,
)
from system_one.validation import validate_questions


BENCHMARK_CASES = [
    {
        "id": "case_1_support_triage",
        "description": "Triagem e Roteamento de Chamado Crítico",
        "state": {
            "ticket_id": "TCK-9921",
            "user_plan": "Enterprise",
            "message": "Nossa API de pagamentos está retornando erro 500 para todos os clientes há 40 minutos! Precisamos de intervenção imediata.",
            "history": "3 chamados resolvidos este mês",
        },
        "questions": {
            "department": Choice(
                instructions="Qual departamento deve receber este ticket?",
                options=[
                    "Engenharia_Backend",
                    "Financeiro",
                    "Suporte_Nivel_1",
                    "Vendas",
                ],
            ),
            "is_critical_outage": Noul(
                instructions="Representa uma indisponibilidade crítica com impacto de receita?"
            ),
            "severity_level": Score(
                instructions="Qual o nível de severidade operacional?",
                levels=["P1_Critico", "P2_Alto", "P3_Medio", "P4_Baixo"],
            ),
        },
    },
    {
        "id": "case_2_content_guardrail",
        "description": "Verificação de Guardrail e Fraude em Mensagem",
        "state": {
            "transaction_id": "TX-4401",
            "amount": 9500.00,
            "user_account_age_days": 1,
            "message_note": "Por favor transfira urgente para a conta externa sem checagem de 2FA.",
        },
        "questions": {
            "risk_verdict": Choice(
                instructions="Decisão de aprovação da transação:",
                options=["Aprovado", "Revisao_Manual", "Bloqueado_Suspeita_Fraude"],
            ),
            "requires_escalation": Noul(
                instructions="Requer escalonamento imediato para equipe de risco?"
            ),
            "risk_score": Score(
                instructions="Nível de risco detectado:",
                levels=[
                    "Risco_Minimo",
                    "Risco_Moderado",
                    "Alto_Risco",
                    "Risco_Critico",
                ],
            ),
        },
    },
    {
        "id": "case_3_lead_qualification",
        "description": "Qualificação Automática de Lead B2B",
        "state": {
            "lead_name": "Tech Corp",
            "company_size": "500-1000",
            "interest": "Estamos avaliando trocar nossa infraestrutura atual por um contrato anual de US$ 50k",
            "decision_maker": True,
        },
        "questions": {
            "lead_tier": Choice(
                instructions="Classificação do Lead:",
                options=[
                    "Tier_1_Enterprise",
                    "Tier_2_MidMarket",
                    "Tier_3_SMB",
                    "Desqualificado",
                ],
            ),
            "ready_for_demo": Noul(
                instructions="O lead tem fit imediato para agendamento de demonstração?"
            ),
            "budget_confidence": Score(
                instructions="Nível de maturidade e clareza de orçamento:",
                levels=["Sem_Orcamento", "Indefinido", "Viavel", "Confirmado_Alto"],
            ),
        },
    },
]


def load_cases(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("Dataset must be a non-empty JSON array")
    factories = {"choice": Choice, "noul": Noul, "score": Score}
    cases, ids = [], set()
    for row in raw:
        if (
            not isinstance(row, dict)
            or not {"id", "state", "questions", "expected"} <= row.keys()
        ):
            raise ValueError("Each case needs id, state, questions, and expected")
        if not isinstance(row["id"], str) or not row["id"].strip() or row["id"] in ids:
            raise ValueError("Case IDs must be unique non-empty strings")
        ids.add(row["id"])
        if not isinstance(row["questions"], dict):
            raise ValueError("questions must be an object")
        questions = {}
        for key, definition in row["questions"].items():
            if not isinstance(definition, dict):
                raise ValueError("Question definitions must be objects")
            definition = dict(definition)
            kind = definition.pop("type", None)
            if kind not in factories:
                raise ValueError("Unknown question type")
            try:
                questions[key] = factories[kind](**definition)
            except TypeError as exc:
                raise ValueError("Invalid question definition") from exc
        validate_questions(questions)
        expected = row["expected"]
        if not isinstance(expected, dict) or set(expected) != set(questions):
            raise ValueError("expected must label every question exactly once")
        for key, question in questions.items():
            label = expected[key]
            if isinstance(question, Noul):
                if type(label) is not bool:
                    raise ValueError("Noul ground truth must be a boolean")
            else:
                allowed = (
                    question.options
                    if isinstance(question, Choice)
                    else question.levels
                )
                if not isinstance(label, str) or label not in allowed:
                    raise ValueError("Ground truth must match a permitted option")
        cases.append({**row, "questions": questions})
    return cases


def run_benchmark(client, cases=None, repeats=1):
    if type(repeats) is not int or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    cases = BENCHMARK_CASES if cases is None else cases
    if not cases:
        raise ValueError("At least one benchmark case is required")
    serializable = [
        {**case, "questions": {k: asdict(q) for k, q in case["questions"].items()}}
        for case in cases
    ]
    digest = hashlib.sha256(
        json.dumps(serializable, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    records, latencies, brier = [], [], []
    correct, labeled, attempted_labels, valid, invalid, failures = 0, 0, 0, 0, 0, 0
    token_fields = ("input_tokens", "output_tokens", "total_tokens")
    reported_tokens = {field: [] for field in token_fields}
    for repeat in range(repeats):
        for case in cases:
            record = {"case_id": case["id"], "repeat": repeat + 1}
            expected = case.get("expected", {})
            attempted_labels += len(expected)
            try:
                result = client.evaluate(case["state"], case["questions"])
            except (InvalidResponseError, ProviderError) as exc:
                invalid += isinstance(exc, InvalidResponseError)
                failures += isinstance(exc, ProviderError)
                record.update(status="error", error_type=type(exc).__name__)
            else:
                valid += 1
                latencies.append(result.metrics.latency_ms)
                for field in token_fields:
                    value = getattr(result.metrics, field)
                    if value is not None:
                        reported_tokens[field].append(value)
                record.update(
                    status="ok",
                    metrics=asdict(result.metrics),
                    answers={
                        key: asdict(answer) for key, answer in result.answers.items()
                    },
                )
                for key, label in expected.items():
                    value = result.answers[key].value
                    labeled += 1
                    if isinstance(case["questions"][key], Noul):
                        correct += (value >= 0.5) == label
                        brier.append((value - int(label)) ** 2)
                    else:
                        correct += value == label
            records.append(record)
    ordered = sorted(latencies)
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "package_version": __version__,
        "python_version": platform.python_version(),
        "provider": client.provider,
        "model": client.model,
        "response_mode": client.capabilities.response_mode,
        "timeout": client.timeout,
        "max_attempts": client.max_retries,
        "use_logprobs": client.use_logprobs,
        "temperature": 0.0,
        "dataset_sha256": digest,
        "unique_cases": len(cases),
        "repeats": repeats,
        "requests": len(records),
        "valid_responses": valid,
        "invalid_responses": invalid,
        "provider_failures": failures,
        "invalid_response_rate": invalid / len(records),
        "successful_latency_p50_ms": statistics.median(ordered) if ordered else None,
        "successful_latency_p95_ms": ordered[math.ceil(0.95 * len(ordered)) - 1]
        if ordered
        else None,
        # Complete totals remain unknown if even one successful response omitted a field.
        **{
            f"successful_{field}": sum(values)
            if valid and len(values) == valid
            else None
            for field, values in reported_tokens.items()
        },
        **{
            f"reported_{field}": sum(values) if values else None
            for field, values in reported_tokens.items()
        },
        "token_usage_coverage": {
            field: {
                "reported_responses": len(values),
                "missing_responses": valid - len(values),
                "fraction": len(values) / valid if valid else None,
            }
            for field, values in reported_tokens.items()
        },
        "estimated_cost_usd": None,
        "labeled_answers": labeled,
        "attempted_labeled_answers": attempted_labels,
        "accuracy_on_valid_answers": correct / labeled if labeled else None,
        "correct_over_attempted_labels": correct / attempted_labels
        if attempted_labels
        else None,
        "noul_brier_score": statistics.mean(brier) if brier else None,
        "records": records,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider", choices=["gemini", "groq", "openai", "ollama"], default="gemini"
    )
    parser.add_argument("--model")
    parser.add_argument("--dataset", help="JSON array of independently labeled cases")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--output", type=Path, help="Write a machine-readable JSON report"
    )
    args = parser.parse_args(argv)
    try:
        cases = load_cases(args.dataset) if args.dataset else BENCHMARK_CASES
        report = run_benchmark(
            SystemOneClient(provider=args.provider, model=args.model),
            cases,
            args.repeat,
        )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 1 if report["invalid_responses"] or report["provider_failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
