import os
from typing import Any, Dict, List, Optional, Tuple, Type

from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from fast_graphrag._llm._base import BaseLLMService
from fast_graphrag._utils import logger


class LangChainLLMService(BaseLLMService):
    """
    A class to interact with LangChain's LLM models, extending the BaseLLMService.

    This class provides a way to send prompts to LangChain models and receive responses.
    """

    def __init__(self, model_name: Optional[str] = "gpt-3.5-turbo-1106", openai_api_key: Optional[str] = None):
        """
        Initialize the LangChainLLMService with a specific model.

        Args:
            model_name (Optional[str]): The name of the LangChain model to use. Defaults to "gpt-3.5-turbo-1106".
            openai_api_key (Optional[str]): The OpenAI API key. If None, it will be read from the OPENAI_API_KEY environment variable.
        """
        if openai_api_key is None:
            openai_api_key = os.environ.get("OPENAI_API_KEY")
        if openai_api_key is None:
            raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable or pass it as an argument.")
        super().__init__(model=model_name)
        self._llm = ChatOpenAI(model_name=model_name, openai_api_key=openai_api_key, temperature=0)

    def get_model_name(self) -> str:
        """
        Get the name of the model.

        Returns:
            str: The name of the model.
        """
        return self._llm.model_name

    async def _send_prompt(
        self,
        prompt_messages: List[BaseMessage],
        response_model: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Send a prompt to the LangChain LLM and return the response.

        Args:
            prompt_messages (List[BaseMessage]): The list of messages to send as a prompt.
            response_model (Optional[Type[BaseModel]]): The response model to parse the response. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple[Any, Dict[str, Any]]: The response from the LLM and the associated metadata.

        Raises:
            ValueError: If the response model is not None and the response cannot be parsed.
        """
        try:
            logger.debug(f"Sending prompt to LangChain: {prompt_messages}")
            llm_response = self._llm(prompt_messages)
            logger.debug(f"Received response from LangChain: {llm_response}")

            if response_model is None:
                return llm_response.content, {}
            else:
                parsed_response = response_model.model_validate_json(llm_response.content)
                return parsed_response, {}

        except Exception as e:
            logger.error(f"Error sending prompt to LangChain: {e}")
            raise