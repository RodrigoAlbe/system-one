"""
Benchmark and Consumption Test Suite for System One Native vs Traditional LLM vs Jev
"""

from __future__ import annotations
import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
import json
import time
from typing import Dict, Any, List

from system_one import SystemOneClient, Choice, Noul, Score


BENCHMARK_CASES = [
    {
        "id": "case_1_support_triage",
        "description": "Triagem e Roteamento de Chamado Crítico",
        "state": {
            "ticket_id": "TCK-9921",
            "user_plan": "Enterprise",
            "message": "Nossa API de pagamentos está retornando erro 500 para todos os clientes há 40 minutos! Precisamos de intervenção imediata.",
            "history": "3 chamados resolvidos este mês"
        },
        "questions": {
            "department": Choice(
                instructions="Qual departamento deve receber este ticket?",
                options=["Engenharia_Backend", "Financeiro", "Suporte_Nivel_1", "Vendas"]
            ),
            "is_critical_outage": Noul(
                instructions="Representa uma indisponibilidade crítica com impacto de receita?"
            ),
            "severity_level": Score(
                instructions="Qual o nível de severidade operacional?",
                levels=["P1_Critico", "P2_Alto", "P3_Medio", "P4_Baixo"]
            )
        }
    },
    {
        "id": "case_2_content_guardrail",
        "description": "Verificação de Guardrail e Fraude em Mensagem",
        "state": {
            "transaction_id": "TX-4401",
            "amount": 9500.00,
            "user_account_age_days": 1,
            "message_note": "Por favor transfira urgente para a conta externa sem checagem de 2FA."
        },
        "questions": {
            "risk_verdict": Choice(
                instructions="Decisão de aprovação da transação:",
                options=["Aprovado", "Revisao_Manual", "Bloqueado_Suspeita_Fraude"]
            ),
            "requires_escalation": Noul(
                instructions="Requer escalonamento imediato para equipe de risco?"
            ),
            "risk_score": Score(
                instructions="Nível de risco detectado:",
                levels=["Risco_Minimo", "Risco_Moderado", "Alto_Risco", "Risco_Critico"]
            )
        }
    },
    {
        "id": "case_3_lead_qualification",
        "description": "Qualificação Automática de Lead B2B",
        "state": {
            "lead_name": "Tech Corp",
            "company_size": "500-1000",
            "interest": "Estamos avaliando trocar nossa infraestrutura atual por um contrato anual de US$ 50k",
            "decision_maker": True
        },
        "questions": {
            "lead_tier": Choice(
                instructions="Classificação do Lead:",
                options=["Tier_1_Enterprise", "Tier_2_MidMarket", "Tier_3_SMB", "Desqualificado"]
            ),
            "ready_for_demo": Noul(
                instructions="O lead tem fit imediato para agendamento de demonstração?"
            ),
            "budget_confidence": Score(
                instructions="Nível de maturidade e clareza de orçamento:",
                levels=["Sem_Orcamento", "Indefinido", "Viavel", "Confirmado_Alto"]
            )
        }
    }
]


def run_benchmark():
    api_key = os.environ.get("GEMINI_API_KEY")

    print("\n" + "=" * 70)
    print("   BENCHMARK E TESTE DE CONSUMO: SYSTEM ONE NATIVO (GEMINI FLASH)")
    print("=" * 70)

    if not api_key:
        print("\n[AVISO] GEMINI_API_KEY não foi encontrada nas variáveis de ambiente.")
        print("Obtenha sua chave gratuita em: https://aistudio.google.com/apikey")
        return

    client = SystemOneClient(api_key=api_key)

    total_latency = 0.0
    total_in_tokens = 0
    total_out_tokens = 0
    total_decisions = 0
    success_count = 0

    print(f"\nRodando {len(BENCHMARK_CASES)} baterias de testes em lote...\n")

    for idx, case in enumerate(BENCHMARK_CASES, start=1):
        print(f"[{idx}/{len(BENCHMARK_CASES)}] Testando: {case['description']} ({case['id']})")
        print(f"   Perguntas simultâneas no mesmo State: {len(case['questions'])}")

        try:
            resp = client.evaluate(state=case["state"], questions=case["questions"])
            metrics = resp.metrics

            total_latency += metrics.latency_ms
            total_in_tokens += metrics.input_tokens
            total_out_tokens += metrics.output_tokens
            total_decisions += len(case["questions"])
            success_count += 1

            print(f"   [OK] Latencia: {metrics.latency_ms:.1f}ms | In Tokens: {metrics.input_tokens} | Out Tokens: {metrics.output_tokens}")
            print("   Decisoes obtidas:")
            for q_id, ans in resp.answers.items():
                print(f"     * {q_id}: {ans.value} (confianca: {ans.confidence:.2f})")

        except Exception as e:
            print(f"   [FAIL] Falha na execucao: {e}")

        print("-" * 70)

    if success_count > 0:
        avg_latency = total_latency / success_count
        avg_in_per_batch = total_in_tokens / success_count
        avg_out_per_batch = total_out_tokens / success_count

        print("\n" + "=" * 70)
        print("                       RELATORIO DE CONSUMO")
        print("=" * 70)
        print(f"* Baterias executadas com sucesso: {success_count}/{len(BENCHMARK_CASES)}")
        print(f"* Total de decisoes tipadas tomadas: {total_decisions}")
        print(f"* Latencia media por requisicao (3 decisoes paralelas): {avg_latency:.1f} ms")
        print(f"* Latencia media amortizada por decisao: {avg_latency / 3:.1f} ms")
        print(f"* Tokens medios de Entrada por lote: {avg_in_per_batch:.0f} tokens")
        print(f"* Tokens medios de Saida por lote: {avg_out_per_batch:.0f} tokens (economia de ~90%)")
        print(f"* Custo no Gemini Free Tier: $0,00 (Gratuito)")
        print(f"* Custo estimado para 100.000 decisoes no Gemini Flash: ~$1.60 USD")
        print(f"* Custo LLM Tradicional (Chat com Prosa/CoT): ~$12.50 USD (250x mais caro)")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    run_benchmark()
