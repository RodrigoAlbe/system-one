"""Public errors callers can route to retry or manual review."""


class InvalidResponseError(ValueError):
    """The provider did not return a complete, valid set of answers."""


class ProviderRefusalError(InvalidResponseError):
    """The provider explicitly refused or blocked the evaluation."""


class IncompleteResponseError(InvalidResponseError):
    """Generation stopped before a complete answer was available."""


class ProviderError(RuntimeError):
    """A transport error or unsuccessful HTTP response prevented evaluation."""
