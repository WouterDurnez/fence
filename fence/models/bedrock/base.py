"""
Base class for Bedrock foundation models
"""

import logging
from typing import Iterator

import boto3

from fence.models.base import LLM, InvokeMixin, MessagesMixin, StreamMixin
from fence.templates.messages import Messages

logger = logging.getLogger(__name__)


class BedrockBase(LLM, MessagesMixin, StreamMixin, InvokeMixin):
    """Base class for Bedrock foundation models"""

    inference_type = "bedrock"

    ###############
    # Constructor #
    ###############

    def __init__(
        self,
        source: str | None = None,
        **kwargs,
    ):
        """
        Initialize a Bedrock model

        :param str source: An indicator of where (e.g., which feature) the model is operating from. Useful to pass to the logging callback
        :param **kwargs: Additional keyword arguments
        """
        super().__init__(**kwargs)
        self.source = source

        # Model parameters
        self.model_kwargs = {
            "temperature": kwargs.get("temperature", 0.01),
            "maxTokens": kwargs.get("max_tokens", 2048),
            "topP": kwargs.get("top_p", 0.9),
        }

        # AWS
        self.region = kwargs.get("region", "eu-central-1")
        self.client = boto3.client("bedrock-runtime", self.region)

    def invoke(self, prompt: str | Messages, **override_kwargs) -> str:
        """
        Call the model with the given prompt
        :param prompt: text to feed the model
        :param override_kwargs: Additional keyword arguments to override the default model kwargs
        :return: response
        """
        return self._invoke(prompt=prompt, stream=False, **override_kwargs)

    def stream(self, prompt: str | Messages, **override_kwargs) -> Iterator[str]:
        """
        Stream the model response with the given prompt.
        :param prompt: text to feed the model
        :param override_kwargs: Additional keyword arguments to override the default model kwargs
        :return: stream of responses
        """
        yield from self._invoke(prompt=prompt, stream=True, **override_kwargs)

    def _invoke(self, prompt: str | Messages, stream: bool, **override_kwargs):
        """
        Centralized method to handle both invoke and stream.
        """
        # Check if the prompt is valid
        self._check_if_prompt_is_valid(prompt=prompt)

        # Prepare the request parameters with override kwargs
        invoke_params = self._prepare_request(prompt, override_kwargs)

        # Invoke the model
        if stream:
            return self._handle_stream(invoke_params)
        else:
            return self._handle_invoke(invoke_params)

    def _prepare_request(self, prompt: str | Messages, override_kwargs: dict) -> dict:
        """
        Prepares the request parameters for the Bedrock API.

        :param prompt: text or Messages object to format
        :param override_kwargs: Additional keyword arguments to override model kwargs
        :return: request parameters dictionary
        """
        messages = self._format_prompt(prompt)

        # Compute effective inference config by merging default and override kwargs
        inference_config = {**self.model_kwargs, **override_kwargs}
        invoke_params = {
            "modelId": self.model_id,
            "messages": messages["messages"],
            "inferenceConfig": inference_config,
        }
        if "system" in messages:
            invoke_params["system"] = messages["system"]
        return invoke_params

    def _handle_invoke(self, invoke_params: dict) -> str:
        """
        Handles synchronous invocation.

        :param invoke_params: parameters for the Bedrock API
        :return: completion text
        """
        try:
            # Call the Bedrock API
            response = self.client.converse(**invoke_params)

            # Process the response
            completion, metrics = self._process_response(response)

            # Log the metrics
            self._log_callback(**metrics)

            # Return the response
            return completion

        except Exception as e:
            raise ValueError(f"Error raised by bedrock service: {e}")

    def _handle_stream(self, invoke_params: dict) -> Iterator[str]:
        """
        Handles streaming invocation.

        :param invoke_params: parameters for the Bedrock API
        :return: stream of responses
        """
        try:
            # Call the Bedrock streaming API
            response = self.client.converse_stream(**invoke_params)

            # Process the streaming response
            stream = response.get("stream")
            if not stream:
                raise ValueError("No stream found in response")

            # Stream the response
            for event in stream:
                # If the event is a message start, print the role
                if "messageStart" in event:
                    logger.debug(
                        f"Message started. Role: {event['messageStart']['role']}"
                    )

                # Yield the completion text if available
                if "contentBlockDelta" in event:
                    text = event["contentBlockDelta"]["delta"]["text"]
                    logger.debug(f"Streaming chunk: {text}")
                    yield text

                # If the event is a message stop, print the stop reason
                if "messageStop" in event:
                    logger.debug(
                        f"Message stopped. Reason: {event['messageStop']['stopReason']}"
                    )

                if "metadata" in event:
                    self._log_callback(**self._extract_metrics(event["metadata"]))

        except Exception as e:
            raise ValueError(f"Error in streaming response from Bedrock service: {e}")

    def _process_response(self, response: dict) -> tuple[str, dict]:
        """
        Extracts completion text and metrics from the response.
        """
        response_body = response.get("output", {}).get("message", {})
        completion = response_body.get("content", [{}])[0].get("text", "")
        metrics = self._extract_metrics(response.get("usage", {}))
        return completion, metrics

    def _extract_metrics(self, metadata: dict) -> dict:
        """
        Extracts relevant metrics from the metadata.
        """
        return {
            "input_token_count": metadata.get("inputTokens", 0),
            "output_token_count": metadata.get("outputTokens", 0),
            "total_token_count": metadata.get("totalTokens", 0),
            "latency_ms": metadata.get("metrics", {}).get("latencyMs"),
        }

    def _format_prompt(self, prompt: str | Messages) -> dict:
        """
        Format the prompt for the Bedrock API

        :param prompt: text or Messages object to format
        :return: formatted messages dictionary
        """
        # Format prompt: Claude3 models expect a list of messages, with content, roles, etc.
        # However, if we receive a string, we will format it as a single user message for ease of use.
        if isinstance(prompt, Messages):
            messages = prompt.model_dump_bedrock_converse()
            # Strip `type` key from each message
            messages["messages"] = [
                {k: v for k, v in message.items() if k != "type"}
                for message in messages["messages"]
            ]
        elif isinstance(prompt, str):
            messages = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "text": prompt,
                            }
                        ],
                    }
                ]
            }
        else:
            raise ValueError("Prompt must be a string or a list of messages")
        return messages
