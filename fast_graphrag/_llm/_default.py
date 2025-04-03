"""Default LLM and Embedding Service."""
from typing import List

import numpy as np
import numpy.typing as npt
from ._llm_openai import OpenAILLMService, OpenAIEmbeddingService


class DefaultLLMService(OpenAILLMService):
    """Default LLM service.

    Uses OpenAI as the default LLM service.
    """

    def get_model_name(self) -> str:
        """Get the name of the LLM model.

        Returns:
            The name of the LLM model.
        """
        return "Default (OpenAI)"


class DefaultEmbeddingService(OpenAIEmbeddingService):
    """Default embedding service.

    Uses OpenAI as the default embedding service.
    """

    async def _encode(self, texts: List[str]) -> npt.NDArray[np.float32]:
        """Encode a list of texts into embeddings.

        Args:
            texts: The texts to encode.
        """
        return await super()._encode(texts)
