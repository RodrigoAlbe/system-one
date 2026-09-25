# System One Architecture Cookbooks

Patterns and production blueprints for structured decision-making.

---

## 1. Triage & Fan-out Routing

Instead of routing messages using large generative LLM prompts, evaluate classification, urgency, and destination simultaneously:

```python
from system_one import SystemOneClient, Choice, Noul, Score

client = SystemOneClient()

state = {
    "source": "slack_alert",
    "service": "auth-service",
    "error_rate": "14.2%",
    "message": "Redis connection pool exhausted"
}

decisions = client.evaluate(
    state=state,
    questions={
        "team": Choice(instructions="Responsible team:", options=["Infra", "Backend_Auth", "Database"]),
        "page_oncall": Noul(instructions="Does this exceed emergency threshold to page on-call engineer?"),
        "incident_priority": Score(instructions="Incident level:", levels=["SEV_1", "SEV_2", "SEV_3"])
    }
)

if decisions.answers["page_oncall"].value > 0.8:
    print(f"Paging {decisions.answers['team'].value} with priority {decisions.answers['incident_priority'].value}")
```

---

## 2. Guardrails & Content Safety Filter

Pre-filter user inputs before feeding them to expensive System 2 reasoning models (saves ~95% of generation costs on harmful or off-topic prompts):

```python
user_prompt = "How can I bypass the software licensing check?"

res = client.evaluate(
    state=user_prompt,
    questions={
        "is_safe": Noul(instructions="Is this prompt safe and compliant with policy?"),
        "intent_category": Choice(
            instructions="Primary intent:",
            options=["Benign_Coding", "Security_Research", "Exploit_Attempt", "Other"]
        )
    }
)

if res.answers["is_safe"].value < 0.3:
    raise ValueError(f"Blocked: detected potential exploit attempt ({res.answers['intent_category'].value})")
```

---

## 3. Lead Qualification & Scoring

Turn unstructured inbound sales form submissions into structured pipeline data:

```python
lead_data = {
    "company": "Acme Inc",
    "headcount": "250",
    "annual_revenue": "$20M",
    "note": "Looking to replace our legacy CRM next month with a team of 40 reps."
}

res = client.evaluate(
    state=lead_data,
    questions={
        "tier": Choice(instructions="Target tier:", options=["Enterprise", "Mid_Market", "SMB"]),
        "ready_for_sales_call": Noul(instructions="Is this lead ready for direct sales contact?"),
        "budget_clarity": Score(instructions="Budget clarity:", levels=["Low", "Medium", "Confirmed"])
    }
)
```
