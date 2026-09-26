"""Provider-specific request formats and response envelopes.

Automatic capability selection is deliberately conservative for custom models.
See README for the documented provider contracts and explicit overrides.
"""

from dataclasses import dataclass

from .errors import InvalidResponseError, IncompleteResponseError, ProviderRefusalError


@dataclass(frozen=True)
class ProviderCapabilities:
    response_mode: str
    logprobs: bool


def capabilities(provider, model, response_mode="auto"):
    if response_mode not in ("auto", "json_schema", "json_object"):
        raise ValueError("response_mode must be auto, json_schema, or json_object")
    mode = "json_object"
    if provider in ("gemini", "ollama"):
        mode = "json_schema"
    elif provider == "openai" and model in {
        "gpt-4o-mini",
        "gpt-4o-mini-2024-07-18",
        "gpt-4o-2024-08-06",
    }:
        mode = "json_schema"
    elif provider == "groq" and model in {"openai/gpt-oss-20b", "openai/gpt-oss-120b"}:
        mode = "json_schema"
    return ProviderCapabilities(
        mode if response_mode == "auto" else response_mode,
        provider in ("openai", "ollama"),
    )


def build_request(
    provider, model, base_url, api_key, schema, prompt, caps, use_logprobs
):
    if provider != "ollama" and not api_key:
        raise ValueError(f"An API key is required for {provider}")
    if use_logprobs and not caps.logprobs:
        raise ValueError(f"Diagnostic logprobs are not enabled for {provider}")
    base_url = base_url.rstrip("/")
    if provider == "gemini":
        config = {"temperature": 0.0, "responseMimeType": "application/json"}
        if caps.response_mode == "json_schema":
            config["responseJsonSchema"] = schema
        return (
            f"{base_url}/{model}:generateContent",
            {"x-goog-api-key": api_key, "Content-Type": "application/json"},
            {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": config},
        )
    response_format = {"type": "json_object"}
    if caps.response_mode == "json_schema":
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "system_one_answers",
                "strict": True,
                "schema": schema,
            },
        }
    payload = {
        "model": model,
        "temperature": 0.0,
        "response_format": response_format,
        "messages": [
            {
                "role": "system",
                "content": "Evaluate the supplied data. Return only JSON matching the requested schema.",
            },
            {"role": "user", "content": prompt},
        ],
    }
    if use_logprobs:
        payload.update(logprobs=True, top_logprobs=5)
    return (
        f"{base_url}/chat/completions",
        {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        payload,
    )


def extract_content(provider, data):
    """Reject refusals/truncation even if a partial payload happens to be valid JSON."""
    try:
        if not isinstance(data, dict):
            raise InvalidResponseError("Provider response must be an object")
        if provider == "gemini":
            if (data.get("promptFeedback") or {}).get("blockReason"):
                raise ProviderRefusalError("Gemini blocked the prompt")
            candidate = data["candidates"][0]
            reason = candidate.get("finishReason")
            if reason == "MAX_TOKENS":
                raise IncompleteResponseError("Gemini output was truncated")
            if reason not in (None, "STOP"):
                raise ProviderRefusalError("Gemini did not complete the evaluation")
            parts = candidate["content"]["parts"]
            if not isinstance(parts, list):
                raise InvalidResponseError("Gemini content parts must be a list")
            content = "".join(
                part["text"]
                for part in parts
                if not part.get("thought") and "text" in part
            )
            usage = data.get("usageMetadata")
            fields = ("promptTokenCount", "candidatesTokenCount", "totalTokenCount")
        else:
            choice = data["choices"][0]
            message = choice["message"]
            if (
                message.get("refusal")
                or choice.get("finish_reason") == "content_filter"
            ):
                raise ProviderRefusalError("Provider refused the evaluation")
            if choice.get("finish_reason") not in (None, "stop"):
                raise IncompleteResponseError("Provider did not finish a text response")
            content = message["content"]
            usage = data.get("usage")
            fields = ("prompt_tokens", "completion_tokens", "total_tokens")
        if usage is None:
            usage = {}
        if not isinstance(usage, dict):
            raise InvalidResponseError("Token usage must be an object")
        counts = tuple(usage.get(field) for field in fields)
        if any(
            count is not None and (type(count) is not int or count < 0)
            for count in counts
        ):
            raise InvalidResponseError(
                "Token counts must be non-negative integers or null"
            )
        return content, counts
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        raise InvalidResponseError("Malformed provider response envelope") from exc
