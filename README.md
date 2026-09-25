# ⚡ System One Native

> **High-speed, zero-cost, typed AI decisions as programming primitives.**  
> A drop-in open-source alternative to TypeSafe Jev powered by Google Gemini (Free Tier), Groq & Ollama with strict JSON Schema.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Skills.sh](https://img.shields.io/badge/skills.sh-agent--ready-green.svg)](https://skills.sh)
[![Claude Code](https://img.shields.io/badge/Claude%20Code-Plugin-purple.svg)](https://claude.ai)

---

## 🎯 What is System One?

Instead of generating free-form conversational text, **System One** treats AI like deterministic software primitives:
* **`Choice`**: Picks one option from a defined set with calibrated confidence.
* **`Noul`**: Returns a boolean probability float ($0.0 \dots 1.0$) for yes/no conditions.
* **`Score`**: Assigns a position along an ordered scale (e.g. `['Low', 'Medium', 'High']`).

**Result:** Zero markdown parsing, zero regex, 90% fewer output tokens, sub-second decisions, and $0 cost on free-tier providers.

---

## 📊 Live Benchmark & Consumption Report

Real benchmark evaluated across multi-question batching (support routing, fraud guardrail, lead triage):

| Metric | Traditional Chat LLM | **System One Native (Gemini / Groq)** | Jev (TypeSafe AI) |
| :--- | :--- | :--- | :--- |
| **Output Tokens** (per 3-decision batch) | ~400 – 800 tokens | **~58 tokens (90% reduction)** | 0 tokens (unmetered) |
| **Parsing Failure Rate** | High (markdown fences, conversational fluff) | **0% (Enforced by Strict JSON Schema)** | 0% (Typed native) |
| **Free Tier** | Varies / None | **100% Free** (Google AI Studio Free Tier / Ollama) | ❌ No Free Tier (HTTP 402) |
| **Projected Cost** (100k decisions) | ~$12.50 – $25.00 USD | **$0.00** (Free Tier) or ~$1.60 (Paid) | ~$0.35 USD (paid only) |
| **Local / Offline Mode** | Requires heavy setup | **✅ Supported via Ollama** | ❌ Cloud proprietary only |

---

## 🚀 Installation

### 1. As an AI Agent Skill (Antigravity, Cursor, Codex, etc.)
Install into your agent workspace or globally via [skills.sh](https://skills.sh):
```bash
npx skills add RodrigoAlbe/system-one --global
```

### 2. In Claude Code
Install as an official Claude Code plugin:
```bash
claude plugin marketplace add RodrigoAlbe/system-one
claude plugin install system-one
```

### 3. In Any Python Project
```bash
pip install system-one-native
```
*(Or install locally in editable mode: `pip install -e .`)*

---

## 💻 Quickstart

### 1. Batch Parallel Decisions (Recommended)
Ask multiple questions over the same state in **one single request**:

```python
from system_one import SystemOneClient, Choice, Noul, Score

client = SystemOneClient()

state = {
    "ticket_id": "TCK-9921",
    "message": "Payment gateway returning 500 error on checkout for all users!",
    "user_plan": "Enterprise"
}

response = client.evaluate(
    state=state,
    questions={
        "dept": Choice(instructions="Routing department:", options=["Backend", "Billing", "DevOps"]),
        "is_urgent": Noul(instructions="Does this represent an active revenue-impacting outage?"),
        "severity": Score(instructions="Operational severity level:", levels=["P1", "P2", "P3", "P4"])
    }
)

print(response.answers["dept"].value)        # "Backend" (confidence: 0.95)
print(response.answers["is_urgent"].value)   # 1.0 (float probability)
print(response.answers["severity"].value)    # "P1" (confidence: 1.0)
print(f"Output tokens: {response.metrics.output_tokens}") # ~55 tokens!
```

### 2. One-Liner Shortcuts
```python
client = SystemOneClient()

# Boolean check (returns float 0.0 - 1.0)
is_spam = client.noul(email_text, "Is this message unsolicited spam?")

# Categorical choice (returns selected string)
category = client.choice(customer_feedback, "Sentiment", ["Positive", "Neutral", "Negative"])

# Score (returns level string)
priority = client.score(task_desc, "Priority", ["Low", "Normal", "Critical"])
```

### 3. Async / Non-Blocking (FastAPI, Telegram/Discord Bots)
```python
import asyncio
from system_one import SystemOneClient, Noul

async def main():
    client = SystemOneClient()
    res = await client.evaluate_async("log payload", {"alert": Noul("Requires immediate page?")})
    print(res.answers["alert"].value)

asyncio.run(main())
```

---

## 🔌 Supported Providers

System One auto-detects your provider based on your environment variables:

| Provider | Environment Variable | Default Model | Notes |
| :--- | :--- | :--- | :--- |
| **Google Gemini (Default)** | `GEMINI_API_KEY` | `gemini-3-flash-preview` | 100% Free Tier via Google AI Studio |
| **Groq Cloud** | `GROQ_API_KEY` | `llama-3.3-70b-versatile` | Ultra-fast inference (~150ms) |
| **OpenAI** | `OPENAI_API_KEY` | `gpt-4o-mini` | Standard structured outputs |
| **Ollama** | None (Localhost) | `qwen2.5:7b` | Fully offline, zero data leaves machine |

Explicitly select a provider:
```python
client = SystemOneClient(provider="groq")   # or "gemini", "ollama", "openai"
```

---

## 🛠️ CLI Usage

You can also run quick decisions directly in your terminal:
```bash
system-one noul "Customer demands immediate refund" "Is this customer angry?"
system-one choice "Server CPU at 99%" "Action" ScaleRestart Ignore
system-one score "Database disk at 92%" "Severity" Low Medium High Critical
```

---

## 🧪 Running Tests & Benchmarks

```bash
# Run unit tests
pytest

# Run live benchmark (requires GEMINI_API_KEY)
python tests/benchmark.py
```

---

## 📄 License

MIT License. Free for commercial and non-commercial use.
