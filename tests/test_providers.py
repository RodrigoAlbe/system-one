import pytest

from system_one import Choice, Noul, Score, SystemOneClient


@pytest.mark.parametrize(
    "provider,model,mode",
    [
        ("openai", None, "json_schema"),
        ("openai", "custom", "json_object"),
        ("groq", None, "json_object"),
        ("groq", "openai/gpt-oss-20b", "json_schema"),
        ("groq", "openai/gpt-oss-120b", "json_schema"),
        ("ollama", None, "json_schema"),
    ],
)
def test_provider_formats(provider, model, mode):
    client = SystemOneClient(provider=provider, model=model, api_key="test")
    _, _, payload = client._prepare_request("state", {"q": Noul("Urgent?")})
    assert payload["response_format"]["type"] == mode
    assert "logprobs" not in payload
    assert "top_logprobs" not in payload
    assert '"probability"' in payload["messages"][1]["content"]
    if mode == "json_schema":
        schema = payload["response_format"]["json_schema"]
        assert schema["strict"] is True
        assert schema["schema"]["additionalProperties"] is False


def test_gemini_json_schema():
    c = SystemOneClient(provider="gemini", api_key="test")
    url, headers, payload = c._prepare_request("state", {"q": Noul("Urgent?")})
    assert url.endswith(":generateContent")
    assert headers["x-goog-api-key"] == "test"
    schema = payload["generationConfig"]["responseJsonSchema"]
    assert schema["additionalProperties"] is False
    assert payload["generationConfig"]["responseMimeType"] == "application/json"


def test_schema_bounds_and_required_fields():
    c = SystemOneClient(provider="openai", api_key="test")
    schema, _ = c._build_schema_and_prompt(
        "s", {"a": Noul("Q"), "b": Choice(["X", "Y"], "Q"), "c": Score(["L", "H"], "Q")}
    )
    assert schema["required"] == ["a", "b", "c"]
    for item in schema["properties"].values():
        assert item["additionalProperties"] is False
        assert set(item["required"]) == set(item["properties"])
        for prop in item["properties"].values():
            if prop["type"] == "number":
                assert (prop["minimum"], prop["maximum"]) == (0, 1)


@pytest.mark.parametrize("provider", ["gemini", "groq"])
def test_unsupported_logprobs_fail_locally(provider):
    with pytest.raises(ValueError, match="logprobs"):
        SystemOneClient(provider=provider, api_key="test", use_logprobs=True)


@pytest.mark.parametrize("provider", ["openai", "ollama"])
def test_diagnostic_logprobs_are_opt_in(provider):
    c = SystemOneClient(provider=provider, api_key="test", use_logprobs=True)
    _, _, payload = c._prepare_request("s", {"q": Noul("Q")})
    assert payload["logprobs"] is True


@pytest.mark.parametrize("provider", ["gemini", "groq", "openai"])
def test_missing_keys_fail_before_request(monkeypatch, provider):
    monkeypatch.setattr("system_one.client._get_env", lambda _: "")
    c = SystemOneClient(provider=provider)
    with pytest.raises(ValueError, match="API key"):
        c._prepare_request("s", {"q": Noul("Q")})


def test_explicit_response_mode_override():
    c = SystemOneClient(
        provider="openai", model="custom", api_key="test", response_mode="json_schema"
    )
    assert (
        c._prepare_request("s", {"q": Noul("Q")})[2]["response_format"]["type"]
        == "json_schema"
    )
    c = SystemOneClient(provider="gemini", api_key="test", response_mode="json_object")
    assert (
        "responseJsonSchema"
        not in c._prepare_request("s", {"q": Noul("Q")})[2]["generationConfig"]
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"timeout": 0},
        {"timeout": float("nan")},
        {"max_retries": 0},
        {"max_retries": True},
        {"response_mode": "invalid"},
    ],
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        SystemOneClient(provider="openai", api_key="test", **kwargs)
