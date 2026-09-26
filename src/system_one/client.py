"""
Core client implementation for System One with multi-provider support.
"""

from __future__ import annotations
import json
import time
import os
import sys
import asyncio
from typing import Any, Dict, List, Optional, Union
import httpx

from .primitives import (
    Choice,
    Noul,
    Score,
    QuestionResult,
    EvaluationMetrics,
    EvaluationResponse,
)
from .logprobs import (
    softmax,
    entropy_confidence,
    extract_openai_logprobs,
    extract_gemini_logprobs,
)


def _get_env(var_name: str) -> str:
    """Gets environment variable with Windows Registry fallback for new sessions."""
    val = os.environ.get(var_name, "")
    if val:
        return val.strip()
    if sys.platform == "win32":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment") as key:
                reg_val, _ = winreg.QueryValueEx(key, var_name)
                return str(reg_val).strip()
        except Exception:
            pass
    return ""


class SystemOneClient:
    """
    Multi-provider System One Decision Engine.
    Executes typed, parallelized, structured evaluations over application state.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 35.0,
        max_retries: int = 3,
        use_logprobs: bool = True,
    ):
        self.provider = (provider or self._detect_provider()).lower()
        self.timeout = timeout
        self.max_retries = max_retries
        self.use_logprobs = use_logprobs

        if self.provider == "gemini":
            raw_key = api_key or _get_env("GEMINI_API_KEY")
            self.api_key = raw_key.strip()
            self.model = model or "gemini-3.1-flash-lite"
            self.base_url = base_url or "https://generativelanguage.googleapis.com/v1beta/models"

        elif self.provider == "groq":
            raw_key = api_key or _get_env("GROQ_API_KEY")
            self.api_key = raw_key.strip()
            self.model = model or "llama-3.3-70b-versatile"
            self.base_url = base_url or "https://api.groq.com/openai/v1"

        elif self.provider == "openai":
            raw_key = api_key or _get_env("OPENAI_API_KEY")
            self.api_key = raw_key.strip()
            self.model = model or "gpt-4o-mini"
            self.base_url = base_url or "https://api.openai.com/v1"

        elif self.provider == "ollama":
            self.api_key = api_key or "ollama"
            self.model = model or "qwen2.5:7b"
            self.base_url = base_url or "http://localhost:11434/v1"

        else:
            raise ValueError(f"Provedor não suportado: {self.provider}. Use 'gemini', 'groq', 'openai' ou 'ollama'.")

    def _detect_provider(self) -> str:
        if _get_env("GEMINI_API_KEY"):
            return "gemini"
        if _get_env("GROQ_API_KEY"):
            return "groq"
        if _get_env("OPENAI_API_KEY"):
            return "openai"
        return "gemini"

    def _build_schema_and_prompt(
        self,
        state: Union[str, Dict[str, Any], List[Any]],
        questions: Dict[str, Union[Choice, Noul, Score]],
    ) -> tuple[dict, str]:
        properties: Dict[str, Any] = {}
        required: List[str] = []
        questions_desc: List[str] = []

        for q_id, q in questions.items():
            required.append(q_id)
            if q.type == "noul":
                properties[q_id] = {
                    "type": "object",
                    "properties": {
                        "probability": {
                            "type": "number",
                            "description": "Probability between 0.0 (definitely no) and 1.0 (definitely yes)",
                        },
                        "confidence": {
                            "type": "number",
                            "description": "Confidence from 0.0 to 1.0",
                        },
                    },
                    "required": ["probability", "confidence"],
                }
                questions_desc.append(f"- Question ID '{q_id}' [NOUL / Yes-No]: {q.instructions}")

            elif q.type == "choice":
                properties[q_id] = {
                    "type": "object",
                    "properties": {
                        "selected": {
                            "type": "string",
                            "enum": q.options,
                            "description": f"One of: {', '.join(q.options)}",
                        },
                        "confidence": {
                            "type": "number",
                            "description": "Confidence from 0.0 to 1.0",
                        },
                    },
                    "required": ["selected", "confidence"],
                }
                questions_desc.append(
                    f"- Question ID '{q_id}' [CHOICE]: {q.instructions} Options: {q.options}"
                )

            elif q.type == "score":
                properties[q_id] = {
                    "type": "object",
                    "properties": {
                        "level": {
                            "type": "string",
                            "enum": q.levels,
                            "description": f"Assigned level from: {', '.join(q.levels)}",
                        },
                        "confidence": {
                            "type": "number",
                            "description": "Confidence from 0.0 to 1.0",
                        },
                    },
                    "required": ["level", "confidence"],
                }
                questions_desc.append(
                    f"- Question ID '{q_id}' [SCORE]: {q.instructions} Levels: {q.levels}"
                )

        schema = {
            "type": "object",
            "properties": properties,
            "required": required,
        }

        state_formatted = (
            json.dumps(state, ensure_ascii=False, indent=2)
            if isinstance(state, (dict, list))
            else str(state)
        )

        prompt = f"""You are an ultra-fast System One Decision Engine.
Do not provide prose, explanations, thoughts, or justifications.
Evaluate the following STATE strictly against each QUESTION, outputting calibrated probabilities and values.

--- STATE ---
{state_formatted}

--- QUESTIONS ---
{chr(10).join(questions_desc)}
"""
        return schema, prompt

    def _prepare_request(
        self,
        state: Union[str, Dict[str, Any], List[Any]],
        questions: Dict[str, Union[Choice, Noul, Score]],
    ) -> tuple[str, dict, dict]:
        schema, prompt = self._build_schema_and_prompt(state, questions)

        if self.provider == "gemini":
            if not self.api_key:
                raise ValueError("GEMINI_API_KEY não encontrada. Defina a variável de ambiente ou passe api_key.")
            url = f"{self.base_url}/{self.model}:generateContent"
            headers = {
                "x-goog-api-key": self.api_key,
                "Content-Type": "application/json",
            }
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.0,
                    "response_mime_type": "application/json",
                    "response_schema": schema,
                },
            }

        else:
            # OpenAI / Groq / Ollama compatible format
            url = f"{self.base_url}/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.model,
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a deterministic System One Decision Engine. Output JSON matching the requested fields only.",
                    },
                    {"role": "user", "content": prompt},
                ],
            }
            if self.use_logprobs:
                payload["logprobs"] = True
                payload["top_logprobs"] = 5

        return url, headers, payload

    def _parse_response(
        self,
        data: Dict[str, Any],
        questions: Dict[str, Union[Choice, Noul, Score]],
        elapsed_ms: float,
    ) -> EvaluationResponse:
        if self.provider == "gemini":
            usage = data.get("usageMetadata", {})
            prompt_tokens = usage.get("promptTokenCount", 0)
            candidates_tokens = usage.get("candidatesTokenCount", 0)
            total_tokens = usage.get("totalTokenCount", 0)
            content_text = data["candidates"][0]["content"]["parts"][0]["text"]
            logprobs_map = extract_gemini_logprobs(data["candidates"][0])
        else:
            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            candidates_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)
            choice_item = data.get("choices", [{}])[0]
            content_text = choice_item.get("message", {}).get("content", "{}")
            logprobs_map = extract_openai_logprobs(choice_item)

        parsed_answers = json.loads(content_text)

        answers: Dict[str, QuestionResult] = {}
        for q_id, q in questions.items():
            ans_data = parsed_answers.get(q_id, {})
            raw_dist = None

            if q.type == "choice":
                matched_logprobs = {opt: logprobs_map[opt] for opt in q.options if opt in logprobs_map}
                if len(matched_logprobs) >= 1:
                    raw_dist = softmax(matched_logprobs)
                    val = max(raw_dist, key=raw_dist.get)
                    conf = entropy_confidence(raw_dist) if len(raw_dist) > 1 else 1.0
                else:
                    val = str(ans_data.get("selected", "")) if isinstance(ans_data, dict) else str(ans_data)
                    conf = float(ans_data.get("confidence", 1.0)) if isinstance(ans_data, dict) else 1.0

            elif q.type == "noul":
                matched_bool = {k: v for k, v in logprobs_map.items() if k.lower() in ("true", "false", "yes", "no", "1", "0")}
                if any(k.lower() in ("true", "yes", "1") for k in matched_bool) and any(k.lower() in ("false", "no", "0") for k in matched_bool):
                    p_map = softmax(matched_bool)
                    p_true = sum(p for k, p in p_map.items() if k.lower() in ("true", "yes", "1"))
                    p_false = sum(p for k, p in p_map.items() if k.lower() in ("false", "no", "0"))
                    total = p_true + p_false
                    norm_true = round(p_true / total, 4) if total > 0 else 0.5
                    raw_dist = {"true": norm_true, "false": round(1.0 - norm_true, 4)}
                    val = norm_true
                    conf = entropy_confidence(raw_dist)
                else:
                    val = float(ans_data.get("probability", 0.0)) if isinstance(ans_data, dict) else float(ans_data)
                    conf = float(ans_data.get("confidence", 1.0)) if isinstance(ans_data, dict) else 1.0

            elif q.type == "score":
                matched_logprobs = {lvl: logprobs_map[lvl] for lvl in q.levels if lvl in logprobs_map}
                if len(matched_logprobs) >= 1:
                    raw_dist = softmax(matched_logprobs)
                    val = max(raw_dist, key=raw_dist.get)
                    conf = entropy_confidence(raw_dist) if len(raw_dist) > 1 else 1.0
                else:
                    val = str(ans_data.get("level", "")) if isinstance(ans_data, dict) else str(ans_data)
                    conf = float(ans_data.get("confidence", 1.0)) if isinstance(ans_data, dict) else 1.0
            else:
                val = ans_data
                conf = 1.0

            answers[q_id] = QuestionResult(
                question_id=q_id,
                question_type=q.type,
                value=val,
                confidence=conf,
                raw_distribution=raw_dist,
            )

        metrics = EvaluationMetrics(
            latency_ms=round(elapsed_ms, 2),
            input_tokens=prompt_tokens,
            output_tokens=candidates_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=0.0,
            provider=self.provider,
            model=self.model,
        )

        return EvaluationResponse(
            answers=answers,
            metrics=metrics,
            raw_response=data,
        )

    def evaluate(
        self,
        state: Union[str, Dict[str, Any], List[Any]],
        questions: Dict[str, Union[Choice, Noul, Score]],
    ) -> EvaluationResponse:
        """
        Synchronous evaluation with automatic retries on temporary outages/rate limits.
        """
        url, headers, payload = self._prepare_request(state, questions)

        start_time = time.perf_counter()
        response = None
        last_error = None

        for attempt in range(self.max_retries):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(url, headers=headers, json=payload)
                if response.status_code == 200:
                    break
                elif response.status_code in (503, 429):
                    time.sleep(1.5 * (attempt + 1))
                    continue
                else:
                    raise RuntimeError(f"Erro na API {self.provider} ({response.status_code}): {response.text}")
            except httpx.TimeoutException as e:
                last_error = e
                time.sleep(1.0)
                continue

        if response is None or response.status_code != 200:
            if last_error:
                raise RuntimeError(f"Timeout após {self.max_retries} tentativas na API {self.provider}: {last_error}")
            raise RuntimeError(f"Erro na API {self.provider} ({response.status_code if response else 'Sem resposta'}): {response.text if response else ''}")

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return self._parse_response(response.json(), questions, elapsed_ms)

    async def evaluate_async(
        self,
        state: Union[str, Dict[str, Any], List[Any]],
        questions: Dict[str, Union[Choice, Noul, Score]],
    ) -> EvaluationResponse:
        """
        Asynchronous non-blocking evaluation for FastAPI, bots, and background workers.
        """
        url, headers, payload = self._prepare_request(state, questions)

        start_time = time.perf_counter()
        response = None
        last_error = None

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for attempt in range(self.max_retries):
                try:
                    response = await client.post(url, headers=headers, json=payload)
                    if response.status_code == 200:
                        break
                    elif response.status_code in (503, 429):
                        await asyncio.sleep(1.5 * (attempt + 1))
                        continue
                    else:
                        raise RuntimeError(f"Erro na API {self.provider} ({response.status_code}): {response.text}")
                except httpx.TimeoutException as e:
                    last_error = e
                    await asyncio.sleep(1.0)
                    continue

        if response is None or response.status_code != 200:
            if last_error:
                raise RuntimeError(f"Timeout após {self.max_retries} tentativas na API {self.provider}: {last_error}")
            raise RuntimeError(f"Erro na API {self.provider}: {response.text if response else ''}")

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return self._parse_response(response.json(), questions, elapsed_ms)

    # Shortcut convenience methods
    def choice(self, state: Any, instructions: str, options: List[str]) -> str:
        """Quick categorical choice."""
        res = self.evaluate(state, {"q": Choice(instructions=instructions, options=options)})
        return str(res.answers["q"].value)

    def noul(self, state: Any, instructions: str) -> float:
        """Quick boolean probability score (0.0 to 1.0)."""
        res = self.evaluate(state, {"q": Noul(instructions=instructions)})
        return float(res.answers["q"].value)

    def score(self, state: Any, instructions: str, levels: List[str]) -> str:
        """Quick graduated score."""
        res = self.evaluate(state, {"q": Score(instructions=instructions, levels=levels)})
        return str(res.answers["q"].value)
