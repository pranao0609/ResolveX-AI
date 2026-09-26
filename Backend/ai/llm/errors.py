class LLMError(Exception):
    """Base exception for LLM-related failures."""


class LLMTimeoutError(LLMError):
    """Raised when an LLM request exceeds the configured timeout."""


class LLMRateLimitError(LLMError):
    """Raised when the provider rate-limits a request."""


class LLMAuthenticationError(LLMError):
    """Raised when provider authentication fails."""


class LLMProviderError(LLMError):
    """Raised for provider-side failures."""


class LLMInvalidResponseError(LLMError):
    """Raised when the provider returns an unusable response."""


class LLMConfigurationError(LLMError):
    """Raised when the LLM configuration is invalid."""
