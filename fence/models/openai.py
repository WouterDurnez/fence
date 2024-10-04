"""
OpenAI GPT models
"""

import json
import os

import requests

__all__ = ["GPT", "GPT4o", "GPT4", "GPT4omini"]

from fence.models.base import LLM, MessagesMixin, get_log_callback
from fence.templates.messages import Message, Messages
from fence.utils.logger import setup_logging

logger = setup_logging(__name__, log_level="info", serious_mode=False)


class GPTBase(LLM, MessagesMixin):
    """Base class for GPT models"""

    model_id = None
    model_name = None
    inference_type = "openai"

    def __init__(
        self,
        source: str | None = None,
        metric_prefix: str | None = None,
        extra_tags: dict | None = None,
        api_key: str | None = None,
        **kwargs,
    ):
        """
        Initialize a GPT model

        :param str source: An indicator of where (e.g., which feature) the model is operating from.
        :param str|None metric_prefix: Prefix for the metric names
        :param dict|None extra_tags: Additional tags to add to the logging tags
        :param str|None api_key: OpenAI API key
        :param **kwargs: Additional keyword arguments
        """

        super().__init__(
            source=source, metric_prefix=metric_prefix, extra_tags=extra_tags
        )
        # Model parameters
        self.model_kwargs = {
            "temperature": kwargs.get("temperature", 1),
            "max_tokens": kwargs.get("max_tokens", None),
        }

        # Find API key
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", None)
        if not self.api_key:
            raise ValueError(
                "OpenAI API key must be provided, either as an argument or in the environment variable `OPENAI_API_KEY`"
            )

        # OpenAI parameters
        self.url = "https://api.openai.com/v1/chat/completions"

    def invoke(self, prompt: str | Messages, **kwargs) -> str:
        """
        Call the model with the given prompt
        :param prompt: text to feed the model
        :return: response
        """

        # Call the model
        response = self._invoke(prompt=prompt)

        # Get input and output tokens
        input_token_count = response["usage"]["prompt_tokens"]
        output_token_count = response["usage"]["completion_tokens"]

        # Get response completion
        completion = response["choices"][0]["message"]["content"]

        # Get input and output word count
        if isinstance(prompt, str):
            input_word_count = len(prompt.split())
        elif isinstance(prompt, Messages):
            input_word_count = self.count_words_in_messages(prompt)
        else:
            raise ValueError(
                f"Prompt must be a string or a list of messages. Got {type(prompt)}"
            )
        output_word_count = len(completion.split())

        # Log all metrics if a log callback is registered
        if log_callback := get_log_callback():
            prefix = ".".join(
                item for item in [self.metric_prefix, self.source] if item
            )
            log_callback(
                # Add metrics
                {
                    f"{prefix}.invocation": 1,
                    f"{prefix}.input_token_count": input_token_count,
                    f"{prefix}.output_token_count": output_token_count,
                    f"{prefix}.input_word_count": input_word_count,
                    f"{prefix}.output_word_count": output_word_count,
                },
                # Format tags as ['key:value', 'key:value', ...]
                self.logging_tags,
            )

        # Calculate token metrics for the response and send them to Datadog
        return completion

    def _invoke(self, prompt: str | Messages) -> dict:
        """
        Handle the API request to the service
        :param prompt: text to feed the model
        :return: response completion
        """

        # Format prompt, using OpenAIs formatting.
        # However, if we receive a string, we will format it as a single user message for ease of use.
        if isinstance(prompt, Messages):

            # Get messages
            messages = [
                message.model_dump(exclude_none=True) for message in prompt.messages
            ]
            system_message = prompt.system

            # Extract system message
            if system_message:
                messages.insert(0, {"role": "system", "content": system_message})
        elif isinstance(prompt, str):
            messages = [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
        else:
            raise ValueError("Prompt must be a string or a list of messages")

        # Build request body
        request_body = {
            **self.model_kwargs,
            "messages": messages,
            "model": self.model_id,
        }

        logger.debug(f"Request body: {request_body}")

        # Send request
        try:

            # Send request to OpenAI
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }

            # Send request to OpenAI
            response = requests.post(
                url=self.url, headers=headers, data=json.dumps(request_body)
            )

            # Check status code
            if response.status_code != 200:
                raise ValueError(f"Error raised by OpenAI service: {response.text}")

            # Parse response
            response = response.json()

            return response

        except Exception as e:
            raise ValueError(f"Error raised by OpenAI service: {e}")


class GPT(GPTBase):
    """
    GPT model
    """

    def __init__(self, model_id: str, **kwargs):
        """
        Initialize a GPT model
        :param model_id: The model ID
        :param source: An indicator of where (e.g., which feature) the model is operating from.
        :param **kwargs: Additional keyword arguments
        """

        super().__init__(**kwargs)
        self.model_id = self.model_name = model_id


class GPT4o(GPT):
    """
    GPT-4o model
    """

    def __init__(self, **kwargs):
        """
        Initialize a GPT-4o model
        :param source: An indicator of where (e.g., which feature) the model is operating from.
        :param **kwargs: Additional keyword arguments
        """
        super().__init__(model_id="gpt-4o", **kwargs)


class GPT4(GPT):
    """
    GPT-4 model
    """

    def __init__(self, **kwargs):
        """
        Initialize a GPT-4 model
        :param source: An indicator of where (e.g., which feature) the model is operating from.
        :param **kwargs: Additional keyword arguments
        """
        super().__init__(model_id="gpt-4", **kwargs)


class GPT4omini(GPT):
    """
    GPT-4mini model
    """

    def __init__(self, **kwargs):
        """
        Initialize a GPT-4mini model
        :param source: An indicator of where (e.g., which feature) the model is operating from.
        :param **kwargs: Additional keyword arguments
        """
        super().__init__(model_id="gpt-4o-mini", **kwargs)


if __name__ == "__main__":

    # Initialize GPT model
    # gpt = GPT4o(source="test")
    # gpt = GPT(model_id="gpt-4", source="test")
    gpt = GPT4omini(source="test")

    # Test prompt
    prompt = "Hello, how are you today?"

    # Call the model
    response = gpt(prompt)
    logger.info(f"GPT response: {response}")

    # Test with Messages
    messages = Messages(
        system="Respond in a all caps",
        messages=[Message(role="user", content="Hello, how are you today?")],
    )

    # Call the model
    response_rude = gpt(messages)
    logger.info(f"GPT response: {response_rude}")
