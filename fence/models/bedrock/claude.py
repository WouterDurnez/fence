"""
Claude Gen 1/2 models
"""

import json
import logging

import boto3

from fence.models.base import (
    LLM,
    get_log_callback,
    register_log_callback,
    register_log_tags,
)

logger = logging.getLogger(__name__)


class ClaudeBase(LLM):
    """Base class for Claude (gen 1/2) models"""

    inference_type = "bedrock"

    def __init__(
        self,
        source: str | None = None,
        metric_prefix: str | None = None,
        extra_tags: dict | None = None,
        **kwargs,
    ):
        """
        Initialize a Claude model

        :param str source: An indicator of where (e.g., which feature) the model is operating from
        :param str|None metric_prefix: Prefix for the metric names
        :param dict|None extra_tags: Additional tags to add to the logging tags
        :param **kwargs: Additional keyword arguments
        """

        super().__init__(
            source=source, metric_prefix=metric_prefix, extra_tags=extra_tags
        )

        # LLM parameters
        self.model_kwargs = {
            "temperature": kwargs.get("temperature", 0.01),
            "max_tokens_to_sample": kwargs.get("max_tokens_to_sample", 2048),
        }

        # AWS parameters
        self.region = kwargs.get("region", "eu-central-1")
        self.client = boto3.client("bedrock-runtime", self.region)

    def invoke(self, prompt: str, **kwargs) -> str:
        """
        Call the model with the given prompt
        :param prompt: text to feed the model
        :param bare_bones: whether to use the 'Human: ... Assistant: ...' formatting
        :return: response
        """

        # Format response
        claude_prompt = f"\n\nHuman: {prompt}\n\nAssistant: "

        # Call the model
        response = self._invoke(prompt=claude_prompt)

        # Get input and output tokens
        input_token_count = response["ResponseMetadata"]["HTTPHeaders"][
            "x-amzn-bedrock-input-token-count"
        ]
        output_token_count = response["ResponseMetadata"]["HTTPHeaders"][
            "x-amzn-bedrock-output-token-count"
        ]

        # Get response completion
        response_body = json.loads(response.get("body").read().decode())
        completion = response_body.get("completion")

        # Get input and output word count
        input_word_count = len(prompt.split())
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

    def _invoke(self, prompt: str) -> str:
        """
        Handle the API request to the service
        :param prompt: text to feed the model
        :return: response completion
        """

        # Payload for post request to bedrock
        input_body = {**self.model_kwargs, "prompt": prompt}
        body = json.dumps(input_body)

        # Headers
        accept = "application/json"
        contentType = "application/json"

        # Send request
        try:
            response = self.client.invoke_model(
                body=body,
                modelId=self.model_id,
                accept=accept,
                contentType=contentType,
            )

            return response

        except Exception as e:
            raise ValueError(f"Error raised by bedrock service: {e}")


class ClaudeInstant(ClaudeBase):
    """Claude Instant model class"""

    def __init__(self, **kwargs):
        """
        Initialize a Claude Instant model
        :param str source: An indicator of where (e.g., which feature) the model is operating from.
        :param **kwargs: Additional keyword arguments
        """

        super().__init__(**kwargs)

        self.model_id = "anthropic.claude-instant-v1"
        self.model_name = "ClaudeInstant"


class ClaudeV2(ClaudeBase):
    """Claude v2 model class"""

    def __init__(self, **kwargs):
        """
        Initialize a Claude Instant model
        :param str source: An indicator of where (e.g., which feature) the model is operating from.
        :param **kwargs: Additional keyword arguments
        """

        super().__init__(**kwargs)

        self.model_id = "anthropic.anthropic.claude-v2"
        self.model_name = "ClaudeV2"


if __name__ == "__main__":

    # Register logging callback
    register_log_callback(lambda metrics, tags: print(metrics, tags))

    # Register logging tags
    register_log_tags({"team": "data-science-test", "project": "fence"})

    # Initialize Claude Instant model
    claude = ClaudeInstant(
        source="test", metric_prefix="supertest", extra_tags={"test": "test"}
    )

    # Call the model
    response = claude("Hello, how are you?")
