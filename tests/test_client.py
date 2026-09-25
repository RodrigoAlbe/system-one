import pytest
from system_one import SystemOneClient, Choice, Noul, Score


def test_schema_generation():
    client = SystemOneClient(provider="gemini", api_key="dummy_key")
    state = {"text": "hello"}
    questions = {
        "cat": Choice(instructions="Category", options=["A", "B"]),
        "flag": Noul(instructions="Is active?"),
        "lvl": Score(instructions="Level", levels=["1", "2", "3"])
    }

    schema, prompt = client._build_schema_and_prompt(state, questions)
    assert schema["type"] == "object"
    assert "cat" in schema["properties"]
    assert "flag" in schema["properties"]
    assert "lvl" in schema["properties"]
    assert schema["properties"]["cat"]["properties"]["selected"]["enum"] == ["A", "B"]
    assert "hello" in prompt


def test_provider_selection():
    c_gemini = SystemOneClient(provider="gemini", api_key="test")
    assert c_gemini.provider == "gemini"

    c_groq = SystemOneClient(provider="groq", api_key="test")
    assert c_groq.provider == "groq"

    c_ollama = SystemOneClient(provider="ollama")
    assert c_ollama.provider == "ollama"
