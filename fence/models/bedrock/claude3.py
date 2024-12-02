"""
Claude Gen 3 models
"""

import json
import logging

import boto3

from fence.models.base import (
    LLM,
    MessagesMixin,
    get_log_callback,
    register_log_callback,
    register_log_tags,
)
from fence.templates.models import Messages

MODEL_ID_SONNET = "anthropic.claude-3-sonnet-20240229-v1:0"
MODEL_ID_HAIKU = "anthropic.claude-3-haiku-20240307-v1:0"

logger = logging.getLogger(__name__)


class Claude3Base(LLM, MessagesMixin):
    """Base class for Claude (gen 3) models"""

    inference_type = "bedrock"

    def __init__(
        self,
        source: str | None = None,
        full_response: bool = False,
        metric_prefix: str | None = None,
        extra_tags: dict | None = None,
        **kwargs,
    ):
        """
        Initialize a Claude model

        :param str source: An indicator of where (e.g., which feature) the model is operating from.
        :param bool full_response: whether to return the full response or just the TEXT completion
        :param str|None metric_prefix: Prefix for the metric names
        :param dict|None extra_tags: Additional tags to add to the logging tags
        :param **kwargs: Additional keyword arguments
        """

        super().__init__(metric_prefix=metric_prefix, extra_tags=extra_tags)

        self.full_response = full_response

        # LLM parameters
        self.model_kwargs = {
            "temperature": kwargs.get("temperature", 0.01),
            "max_tokens": kwargs.get("max_tokens", 2048),
        }

        # AWS parameters
        self.region = kwargs.get("region", "eu-central-1")
        self.client = boto3.client("bedrock-runtime", self.region)

    def invoke(self, prompt: str | Messages, **kwargs) -> str:
        """
        Call the model with the given prompt
        :param prompt: text to feed the model
        :return: response
        """

        # Call the model
        response = self._invoke(prompt=prompt)

        # Get response completion
        response_body = json.loads(response.get("body").read().decode())

        # Get token counts
        input_token_count = response_body["usage"]["input_tokens"]
        output_token_count = response_body["usage"]["output_tokens"]

        # Get completion
        completion = response_body["content"][0]["text"]

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
            log_args = [
                # Add metrics
                {
                    f"{prefix}.invocation": 1,
                    f"{prefix}.input_token_count": input_token_count,
                    f"{prefix}.output_token_count": output_token_count,
                    f"{prefix}.input_word_count": input_word_count,
                    f"{prefix}.output_word_count": output_word_count,
                },
                # Add tags
                self.logging_tags,
            ]

            # Log metrics
            logger.debug(f"Logging args: {log_args}")
            log_callback(*log_args)

        return completion

    def _invoke(self, prompt: str | Messages) -> dict:
        """
        Handle the API request to the service
        :param prompt: text to feed the model
        :return: response completion
        """

        # Format prompt: Claude3 models expect a list of messages, with content, roles, etc.
        # However, if we receive a string, we will format it as a single user message for ease of use.
        if isinstance(prompt, Messages):
            messages = prompt.model_dump(exclude_none=True)
        elif isinstance(prompt, str):
            messages = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt,
                            }
                        ],
                    }
                ]
            }
        else:
            raise ValueError("Prompt must be a string or a list of messages")

        # Build request body
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            **messages,
            **self.model_kwargs,
        }

        logger.debug(f"Request body: {request_body}")

        # Send request
        try:
            response = self.client.invoke_model(
                modelId=self.model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(request_body),
            )

            # Log invocation
            self.invocation_logging()

            return response

        except Exception as e:
            raise ValueError(f"Error raised by bedrock service: {e}")

    def invocation_logging(self) -> None:
        """
        Log invocation
        :return:
        """
        pass


class ClaudeHaiku(Claude3Base):
    """Claude Haiku model class"""

    def __init__(self, **kwargs):
        """
        Initialize a Claude Haiku model
        :param str source: An indicator of where (e.g., which feature) the model is operating from.
        :param **kwargs: Additional keyword arguments
        """

        super().__init__(**kwargs)

        self.model_id = MODEL_ID_HAIKU
        self.model_name = "ClaudeHaiku"


class ClaudeSonnet(Claude3Base):
    """Claude Sonnet model class"""

    def __init__(self, **kwargs):
        """
        Initialize a Claude Sonnet model
        :param str source: An indicator of where (e.g., which feature) the model is operating from.
        :param **kwargs: Additional keyword arguments
        """

        super().__init__(**kwargs)

        self.model_id = MODEL_ID_SONNET
        self.model_name = "ClaudeSonnet"


class Claude35Sonnet(Claude3Base):
    """Claude Sonnet 3.5 model class"""

    def __init__(self, **kwargs):
        """
        Initialize a Claude Sonnet 3.5 model
        :param str source: An indicator of where (e.g., which feature) the model is operating from.
        :param **kwargs: Additional keyword arguments
        """

        super().__init__(**kwargs)

        self.model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"
        self.model_name = "ClaudeSonnet3.5"


if __name__ == "__main__":

    # Register logging callback
    register_log_callback(lambda metrics, tags: print(metrics, tags))

    # Register logging tags
    register_log_tags({"team": "data-science-test", "project": "fence"})

    # Create an instance of the ClaudeHaiku class
    claude_sonnet = Claude35Sonnet(
        source="test", metric_prefix="yolo", region="us-east-1"
    )

    # Call the invoke method with a prompt
    response = claude_sonnet.invoke(prompt="The sun is shining brightly")
