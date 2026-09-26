import asyncio
import json

import httpx
import pytest

from system_one import SystemOneClient, Noul, InvalidResponseError, ProviderError


VALID = {
    "choices": [
        {
            "message": {"content": '{"q":{"probability":0.8,"confidence":0.9}}'},
            "finish_reason": "stop",
        }
    ]
}


@pytest.fixture
def transport(monkeypatch):
    requests, sleeps = [], []
    sync_cls, async_cls = httpx.Client, httpx.AsyncClient

    def install(outcomes):
        pending = iter(outcomes)

        def handle(request):
            requests.append(request)
            outcome = next(pending)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        mock = httpx.MockTransport(handle)
        monkeypatch.setattr(
            httpx, "Client", lambda **kw: sync_cls(transport=mock, **kw)
        )
        monkeypatch.setattr(
            httpx, "AsyncClient", lambda **kw: async_cls(transport=mock, **kw)
        )
        monkeypatch.setattr("system_one.client.time.sleep", sleeps.append)

        async def sleep(delay):
            sleeps.append(delay)

        monkeypatch.setattr("system_one.client.asyncio.sleep", sleep)
        return requests, sleeps

    return install


def evaluate(async_mode, **kwargs):
    c = SystemOneClient(provider="openai", api_key="test", **kwargs)
    if async_mode:
        return asyncio.run(c.evaluate_async("state", {"q": Noul("Question")}))
    return c.evaluate("state", {"q": Noul("Question")})


@pytest.mark.parametrize("async_mode", [False, True])
def test_success_uses_actual_httpx_transport(transport, async_mode):
    requests, sleeps = transport([httpx.Response(200, json=VALID)])
    result = evaluate(async_mode)
    assert result.answers["q"].value == 0.8
    assert len(requests) == 1
    assert sleeps == []
    payload = json.loads(requests[0].content)
    assert payload["response_format"]["json_schema"]["strict"] is True


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("first", [429, 503, 500, "timeout", "connection"])
def test_transient_failures_retry(transport, async_mode, first):
    if first == "timeout":
        outcome = httpx.ReadTimeout("Timeout")
    elif first == "connection":
        outcome = httpx.ConnectError("Connection")
    else:
        outcome = httpx.Response(first, headers={"Retry-After": "2"})
    requests, sleeps = transport([outcome, httpx.Response(200, json=VALID)])
    assert evaluate(async_mode).answers["q"].value == 0.8
    assert len(requests) == 2
    assert sleeps == [2.0 if isinstance(first, int) else 1.5]


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("status", [400, 401, 403])
def test_permanent_failures_do_not_retry_or_expose_body(transport, async_mode, status):
    requests, sleeps = transport(
        [httpx.Response(status, text="sensitive provider body")]
    )
    with pytest.raises(ProviderError, match=str(status)) as exc:
        evaluate(async_mode)
    assert "sensitive" not in str(exc.value)
    assert len(requests) == 1
    assert sleeps == []


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("failure", ["timeout", "http"])
def test_exhausted_attempts_never_sleep_after_last_failure(
    transport, async_mode, failure
):
    outcomes = [
        httpx.ReadTimeout("timeout") if failure == "timeout" else httpx.Response(429)
        for _ in range(3)
    ]
    requests, sleeps = transport(outcomes)
    with pytest.raises(ProviderError):
        evaluate(async_mode)
    assert len(requests) == 3
    assert sleeps == [1.5, 3.0]


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize(
    "content", [b"not JSON", b"{}", b'{"choices":[{"message":{"content":"{}"}}]}']
)
def test_invalid_responses_fail_atomically_without_retries(
    transport, async_mode, content
):
    requests, sleeps = transport([httpx.Response(200, content=content)])
    with pytest.raises(InvalidResponseError):
        evaluate(async_mode)
    assert len(requests) == 1
    assert sleeps == []


@pytest.mark.parametrize("async_mode", [False, True])
def test_most_recent_error_wins(transport, async_mode):
    transport([httpx.ReadTimeout("old timeout"), httpx.Response(503)])
    with pytest.raises(ProviderError, match="503"):
        evaluate(async_mode, max_retries=2)


def test_async_cancellation_is_not_retried(transport):
    async def run():
        async def cancel(request):
            raise asyncio.CancelledError()

        original = httpx.AsyncClient
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(
                httpx,
                "AsyncClient",
                lambda **kw: original(transport=httpx.MockTransport(cancel), **kw),
            )
            with pytest.raises(asyncio.CancelledError):
                await SystemOneClient(provider="openai", api_key="test").evaluate_async(
                    "s", {"q": Noul("Q")}
                )

    asyncio.run(run())
