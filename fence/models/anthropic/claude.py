"""
Anthropic Claude models
"""

import json
import logging
import os

import requests

from fence.models.base import LLM, MessagesMixin, get_log_callback
from fence.templates.messages import Message, Messages
from fence.utils.logger import setup_logging

logger = logging.getLogger(__name__)


class ClaudeBase(LLM, MessagesMixin):
    """Base class for GPT models"""

    inference_type = "anthropic"

    def __init__(
        self,
        source: str | None = None,
        metric_prefix: str | None = None,
        extra_tags: dict | None = None,
        api_key: str | None = None,
        **kwargs,
    ):
        """
        Initialize a Claude model

        :param str source: An indicator of where (e.g., which feature) the model is operating from.
        :param str|None metric_prefix: Prefix for the metric names
        :param dict|None extra_tags: Additional tags to add to the logging tags
        :param str|None api_key: Anthropic API key
        :param **kwargs: Additional keyword arguments
        """

        super().__init__(
            source=source, metric_prefix=metric_prefix, extra_tags=extra_tags
        )
        # Model parameters
        self.model_kwargs = {
            "temperature": kwargs.get("temperature", 1),
            "max_tokens": kwargs.get("max_tokens", 2048),
        }

        # Find API key
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", None)
        if not self.api_key:
            raise ValueError(
                "Anthropic API key must be provided, either as an argument or in the environment variable `ANTHROPIC_API_KEY`"
            )

        # Anthropic parameters
        self.url = "https://api.anthropic.com/v1/messages"

    def invoke(self, prompt: str | Messages, **kwargs) -> str:
        """
        Call the model with the given prompt
        :param prompt: text to feed the model
        :return: response
        """

        # Prompt should not be empty
        self._check_if_prompt_is_valid(prompt=prompt)

        # Call the model
        response = self._invoke(prompt=prompt)
        logger.debug(f"Response: {response}")

        # Get input and output tokens
        input_token_count = response["usage"]["input_tokens"]
        output_token_count = response["usage"]["output_tokens"]

        # Get response completion
        completion = response["content"][0]["text"]

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

        # Format prompt, using Anthropics formatting.
        # However, if we receive a string, we will format it as a single user message for ease of use.
        if isinstance(prompt, Messages):

            # Format messages
            messages = prompt.model_dump_anthropic()

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

        # Add system prompt if available
        if isinstance(prompt, Messages):
            if prompt.system:
                request_body["system"] = prompt.system

        logger.debug(f"Request body: {request_body}")

        # Send request
        try:

            # Send request to Anthropic
            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            }

            # Send request to Anthropic
            response = requests.post(
                url=self.url, headers=headers, data=json.dumps(request_body)
            )

            # Check status code
            if response.status_code != 200:
                raise ValueError(f"Error raised by Anthropic service: {response.text}")

            # Parse response
            response = response.json()

            return response

        except Exception as e:
            raise ValueError(f"Error raised by Anthropic service: {e}")


class Claude(ClaudeBase):
    """
    Claude model
    """

    def __init__(self, model_id: str, model_name: str, **kwargs):
        """
        Initialize a Claude model
        :param model_id: The model ID
        :param model_name: The model name
        :param source: An indicator of where (e.g., which feature) the model is operating from.
        :param **kwargs: Additional keyword arguments
        """

        self.model_id = model_id
        self.model_name = model_name

        super().__init__(**kwargs)


class Claude35Haiku(Claude):
    """
    Claude 3.5 Haiku model
    """

    def __init__(self, **kwargs):
        """
        Initialize a Claude 3.5 Haiku model
        :param str source: An indicator of where (e.g., which feature) the model is operating from.
        :param **kwargs: Additional keyword arguments
        """

        super().__init__(
            model_id="claude-3-5-haiku-20241022",
            model_name="Claude 3.5 Haiku [Anthropic]",
            **kwargs,
        )


class Claude35Sonnet(Claude):
    """
    Claude 3.5 Sonnet model
    """

    def __init__(self, **kwargs):
        """
        Initialize a Claude 3.5 Sonnet model
        :param str source: An indicator of where (e.g., which feature) the model is operating from.
        :param **kwargs: Additional keyword arguments
        """

        super().__init__(
            model_id="claude-3-5-sonnet-20241022",
            model_name="Claude 3.5 Sonnet [Anthropic]",
            **kwargs,
        )


class Claude4Opus(Claude):
    """
    Claude 4 Opus model - Our most capable and intelligent model yet
    """

    def __init__(self, **kwargs):
        """
        Initialize a Claude 4 Opus model
        :param str source: An indicator of where (e.g., which feature) the model is operating from.
        :param **kwargs: Additional keyword arguments
        """

        super().__init__(
            model_id="claude-opus-4-20250514",
            model_name="Claude 4 Opus [Anthropic]",
            **kwargs,
        )


class Claude4Sonnet(Claude):
    """
    Claude 4 Sonnet model - High-performance model with exceptional reasoning capabilities
    """

    def __init__(self, **kwargs):
        """
        Initialize a Claude 4 Sonnet model
        :param str source: An indicator of where (e.g., which feature) the model is operating from.
        :param **kwargs: Additional keyword arguments
        """

        super().__init__(
            model_id="claude-sonnet-4-20250514",
            model_name="Claude 4 Sonnet [Anthropic]",
            **kwargs,
        )


class Claude37Sonnet(Claude):
    """
    Claude 3.7 Sonnet model - High-performance model with early extended thinking
    """

    def __init__(self, **kwargs):
        """
        Initialize a Claude 3.7 Sonnet model
        :param str source: An indicator of where (e.g., which feature) the model is operating from.
        :param **kwargs: Additional keyword arguments
        """

        super().__init__(
            model_id="claude-3-7-sonnet-20250219",
            model_name="Claude 3.7 Sonnet [Anthropic]",
            **kwargs,
        )


class Claude3Opus(Claude):
    """
    Claude 3 Opus model - Powerful model for complex tasks
    """

    def __init__(self, **kwargs):
        """
        Initialize a Claude 3 Opus model
        :param str source: An indicator of where (e.g., which feature) the model is operating from.
        :param **kwargs: Additional keyword arguments
        """

        super().__init__(
            model_id="claude-3-opus-20240229",
            model_name="Claude 3 Opus [Anthropic]",
            **kwargs,
        )


class Claude3Haiku(Claude):
    """
    Claude 3 Haiku model - Fast and compact model for near-instant responsiveness
    """

    def __init__(self, **kwargs):
        """
        Initialize a Claude 3 Haiku model
        :param str source: An indicator of where (e.g., which feature) the model is operating from.
        :param **kwargs: Additional keyword arguments
        """

        super().__init__(
            model_id="claude-3-haiku-20240307",
            model_name="Claude 3 Haiku [Anthropic]",
            **kwargs,
        )


if __name__ == "__main__":

    setup_logging(log_level="DEBUG")
    # Initialize Claude model
    model = Claude35Haiku()
    # Test with a prompt
    prompt = "Write a haiku about the ocean"

    # Call the model
    response = model(prompt)
    logger.info(f"Claude35Haiku response: {response}")

    # Test with Messages
    messages = Messages(
        system="Respond in a all caps",
        messages=[Message(role="user", content="Hello, how are you today?")],
    )

    # Call the model
    response_rude = model(messages)
    logger.info(f"Claude35Haiku response: {response_rude}")

    # Test Claude 4 Sonnet
    logger.info("Testing Claude 4 Sonnet...")
    claude4_model = Claude4Sonnet()
    response_claude4 = claude4_model("Explain quantum computing in simple terms.")
    logger.info(f"Claude4Sonnet response: {response_claude4}")

    # Test Claude 4 Opus
    logger.info("Testing Claude 4 Opus...")
    claude4_opus = Claude4Opus()
    response_opus = claude4_opus(
        "Write a Python function to calculate fibonacci numbers."
    )
    logger.info(f"Claude4Opus response: {response_opus}")
