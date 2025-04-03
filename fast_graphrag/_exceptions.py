class LLMServiceNoResponseError(Exception):
    """Error raised when the LLM service does not return a response."""


class LLMServiceError(Exception):
    """Base error for LLM service errors."""


class InvalidStorageError(Exception):
    """Error raised when the storage is not valid."""
