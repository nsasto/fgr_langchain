__all__ = [
    "DefaultEmbeddingService",
    "DefaultLLMService",
]

from typing import List, Tuple, Any, Dict, Optional, Union, Callable

from fast_graphrag._types import GTEmbedding, GTHash
from ._base import BaseEmbeddingService, BaseLLMService, format_and_send_prompt as _format_and_send_prompt
from ._default import DefaultEmbeddingService as _DefaultEmbeddingService, DefaultLLMService as _DefaultLLMService


class DefaultLLMService(_DefaultLLMService):
    """Default LLM service implementation."""

    pass


class DefaultEmbeddingService(_DefaultEmbeddingService):
    """Default embedding service implementation."""

    pass


async def format_and_send_prompt(
    prompt_key: str,
    llm: BaseLLMService,
    format_kwargs: Optional[Dict[str, Any]] = None,
    response_model: Optional[Any] = None,
    validation_seed: Optional[int] = None,
    return_first: bool = False,
    functions: Optional[List[Union[Dict[str, Any], Callable]]] = None,
    history: Optional[List[Tuple[str, str]]] = None,
    return_on_failure: Any = None,
) -> Tuple[Any, List[Dict[str, str]]]:
    """Format and send a prompt to the LLM service."""
    return await _format_and_send_prompt(
        prompt_key=prompt_key,
        llm=llm,
        format_kwargs=format_kwargs,
        response_model=response_model,
        validation_seed=validation_seed,
        return_first=return_first,
        functions=functions,
        history=history,
        return_on_failure=return_on_failure,
    )
