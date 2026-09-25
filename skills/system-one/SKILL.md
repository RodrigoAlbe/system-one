---
name: system-one
license: MIT
description: >
  Build high-speed, zero-cost AI-powered decisions with System One: structured,
  calibrated judgments that software can use directly like programming primitives.
  Drop-in open-source alternative to TypeSafe Jev powered by Google Gemini (Free Tier),
  Groq, or local Ollama with strict JSON Schema output. Use whenever a workflow needs
  intent routing, classification, guardrail verification (boolean yes/no), risk scoring,
  or triage instead of expensive, verbose chat LLM pipelines.
---

# Build with System One Native

System One makes units of AI intelligence usable like programming primitives: small,
calibrated judgments you can compose into larger capabilities without generating chat prose.

## Core Primitives

| Primitive | Use Case | Returns |
| :--- | :--- | :--- |
| **`Choice(options, instructions)`** | Pick one option from a defined set | Selected option string (`selected`) with confidence score (`0.0` - `1.0`) |
| **`Noul(instructions)`** | Check if a condition holds (yes/no) | Calibrated probability float (`0.0` - `1.0`) |
| **`Score(levels, instructions)`** | Degree along an ordered scale | Assigned tier string (`level`) with confidence score |

---

## Usage in Code

### 1. Multi-Question Batch Evaluation (1 Single Request)
When you have multiple questions about the same `state`, **always evaluate them together**. They run in parallel in a single LLM call, saving ~90% tokens:

```python
from system_one import SystemOneClient, Choice, Noul, Score

client = SystemOneClient()

state = {
    "ticket_id": "T-102",
    "message": "Payment failed during checkout with code 500.",
    "user_plan": "Enterprise"
}

response = client.evaluate(
    state=state,
    questions={
        "dept": Choice(instructions="Routing department:", options=["Billing", "Backend", "Support"]),
        "is_urgent": Noul(instructions="Does this represent an active revenue-impacting outage?"),
        "severity": Score(instructions="Operational severity:", levels=["P1", "P2", "P3", "P4"])
    }
)

print(response.answers["dept"].value)        # "Backend"
print(response.answers["is_urgent"].value)   # 1.0 (float probability)
print(response.answers["severity"].value)    # "P1"
print(f"Latency: {response.metrics.latency_ms}ms")
```

### 2. Convenience Shortcut Methods

```python
client = SystemOneClient()

# Boolean probability (0.0 to 1.0)
is_fraud = client.noul(transaction, "Is this transaction fraudulent or suspicious?")

# Categorical choice
route = client.choice(email, "Select category", ["Sales", "Billing", "Tech"])

# Graduated scoring
tier = client.score(lead, "Lead qualification", ["Unqualified", "SMB", "Enterprise"])
```

### 3. Asynchronous Non-Blocking Execution (FastAPI / Workers)

```python
import asyncio
from system_one import SystemOneClient, Noul

async def check():
    client = SystemOneClient()
    res = await client.evaluate_async("log payload", {"alert": Noul("Does this require on-call alert?")})
    print(res.answers["alert"].value)

asyncio.run(check())
```

---

## Provider Configuration

System One auto-detects your provider based on available environment variables:

1. **Google Gemini (Default, Free Tier):**
   * Set `GEMINI_API_KEY="your_key"` (free at https://aistudio.google.com/apikey).
   * Models: `gemini-3-flash-preview`, `gemini-flash-latest`.
2. **Groq Cloud (Sub-200ms latency):**
   * Set `GROQ_API_KEY="gsk_..."`.
   * Models: `llama-3.3-70b-versatile`, `llama-3.1-8b-instant`.
3. **Local Ollama (Offline, zero API keys):**
   * Run `ollama run qwen2.5:7b` or `llama3.2`.
   * Initialize: `client = SystemOneClient(provider="ollama")`.

See [cookbooks](./references/cookbooks.md) for complete architectural patterns.
