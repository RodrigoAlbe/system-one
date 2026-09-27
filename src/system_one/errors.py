"""Public errors callers can route to retry or manual review."""


class InvalidResponseError(ValueError):
    """The provider did not return a complete, valid set of answers."""


class ProviderRefusalError(InvalidResponseError):
    """The provider explicitly refused or blocked the evaluation."""


class IncompleteResponseError(InvalidResponseError):
    """Generation stopped before a complete answer was available."""


class ProviderError(RuntimeError):
    """A transport error or unsuccessful HTTP response prevented evaluation."""

    def __init__(
        self,
        message,
        *,
        provider=None,
        status_code=None,
        retryable=False,
        retry_after=None,
        attempts=0,
        request_id=None,
    ):
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.retryable = retryable
        self.retry_after = retry_after
        self.attempts = attempts
        self.request_id = request_id


class EvaluationTimeoutError(ProviderError):
    """The evaluation exhausted its budget or cannot fit the next retry wait."""
