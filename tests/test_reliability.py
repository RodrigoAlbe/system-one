import asyncio

import httpx
import pytest

from system_one import EvaluationTimeoutError, Noul, ProviderError, SystemOneClient
from system_one.logprobs import entropy_confidence, softmax

VALID = {
    "choices": [
        {
            "message": {"content": '{"q":{"probability":0.8,"confidence":0.9}}'},
            "finish_reason": "stop",
        }
    ]
}


@pytest.mark.parametrize("async_mode", [False, True])
def test_long_retry_wait_is_rejected_with_metadata(monkeypatch, async_mode):
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(
            429, headers={"Retry-After": "86400", "x-request-id": "req-test"}
        )

    sync, async_cls = httpx.Client, httpx.AsyncClient
    monkeypatch.setattr(
        httpx, "Client", lambda **kw: sync(transport=httpx.MockTransport(handle), **kw)
    )
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kw: async_cls(transport=httpx.MockTransport(handle), **kw),
    )
    c = SystemOneClient(provider="openai", api_key="test", total_timeout=1)

    async def run():
        async with c:
            await c.evaluate_async("s", {"q": Noul("Q")})

    with pytest.raises(EvaluationTimeoutError) as caught:
        if async_mode:
            asyncio.run(run())
        else:
            with c:
                c.evaluate("s", {"q": Noul("Q")})
    error = caught.value
    assert (error.status_code, error.attempts, error.retry_after) == (429, 1, 86400)
    assert error.retryable and error.request_id == "req-test"
    assert len(calls) == 1


def test_async_deadline_cancels_inflight_request(monkeypatch):
    cancelled = []

    async def handle(request):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(True)

    original = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kw: original(transport=httpx.MockTransport(handle), **kw),
    )

    async def run():
        async with SystemOneClient(
            provider="openai", api_key="test", total_timeout=0.02
        ) as c:
            with pytest.raises(EvaluationTimeoutError) as caught:
                await c.evaluate_async("s", {"q": Noul("Q")})
            assert caught.value.attempts == 1

    asyncio.run(run())
    assert cancelled == [True]


@pytest.mark.parametrize("async_mode", [False, True])
def test_connection_client_reused_and_closed(monkeypatch, async_mode):
    instances = []
    original = httpx.AsyncClient if async_mode else httpx.Client

    def factory(**kw):
        instance = original(
            transport=httpx.MockTransport(lambda req: httpx.Response(200, json=VALID)),
            **kw,
        )
        instances.append(instance)
        return instance

    monkeypatch.setattr(httpx, "AsyncClient" if async_mode else "Client", factory)
    c = SystemOneClient(provider="openai", api_key="test")

    async def run():
        async with c:
            for _ in range(2):
                await c.evaluate_async("s", {"q": Noul("Q")})
        with pytest.raises(RuntimeError, match="closed"):
            await c.evaluate_async("s", {"q": Noul("Q")})

    if async_mode:
        asyncio.run(run())
    else:
        with c:
            for _ in range(2):
                c.evaluate("s", {"q": Noul("Q")})
        with pytest.raises(RuntimeError, match="closed"):
            c.evaluate("s", {"q": Noul("Q")})
    assert len(instances) == 1 and instances[0].is_closed


@pytest.mark.parametrize(
    "p",
    [
        {},
        {"a": 0, "b": 0},
        {"a": -1, "b": 2},
        {"a": float("nan")},
        {"a": float("inf")},
        {"a": True},
        {"a": 0.7, "b": 0.7},
    ],
)
def test_invalid_distribution_rejected(p):
    with pytest.raises(ValueError):
        entropy_confidence(p)


def test_softmax_does_not_round_away_probability_mass():
    p = softmax({str(i): 0 for i in range(30000)})
    assert sum(p.values()) == pytest.approx(1)
    assert entropy_confidence(p) == pytest.approx(0)


@pytest.mark.parametrize("value", [0, -1, True, float("nan"), float("inf"), None])
def test_invalid_total_timeout(value):
    with pytest.raises(ValueError, match="total_timeout"):
        SystemOneClient(total_timeout=value)


def test_sync_expired_response_never_becomes_a_decision(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("system_one.client.time.perf_counter", lambda: clock[0])

    def handle(req):
        clock[0] = 2.0
        return httpx.Response(200, json=VALID)

    original = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kw: original(transport=httpx.MockTransport(handle), **kw),
    )
    with SystemOneClient(provider="openai", api_key="test", total_timeout=1) as c:
        with pytest.raises(EvaluationTimeoutError):
            c.evaluate("s", {"q": Noul("Q")})


def test_permanent_error_metadata_excludes_body(monkeypatch):
    original = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kw: original(
            transport=httpx.MockTransport(
                lambda req: httpx.Response(401, text="secret")
            ),
            **kw,
        ),
    )
    with SystemOneClient(provider="openai", api_key="test") as c:
        with pytest.raises(ProviderError) as caught:
            c.evaluate("s", {"q": Noul("Q")})
    assert caught.value.status_code == 401
    assert not caught.value.retryable
    assert caught.value.attempts == 1
    assert "secret" not in str(caught.value)
