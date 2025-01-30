import logging

import boto3

from fence.models.base import LLM, MessagesMixin, get_log_callback
from fence.templates.messages import Messages

logger = logging.getLogger(__name__)
MODEL_ID_HAIKU = "anthropic.claude-3-haiku-20240307-v1:0"
MODEL_ID_PRO = "amazon.nova-pro-v1:0"


class BedrockBase(LLM, MessagesMixin):
    """Base class for Bedrock foundation models"""

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
        Initialize a Bedrock model

        :param str source: An indicator of where (e.g., which feature) the model is operating from. Useful to pass to the logging callback
        :param bool full_response: whether to return the full response object or just the TEXT completion
        :param str|None metric_prefix: Prefix for the metric names
        :param dict|None extra_tags: Additional tags to add to the logging tags
        :param **kwargs: Additional keyword arguments
        """

        super().__init__(metric_prefix=metric_prefix, extra_tags=extra_tags)

        self.source = source
        self.full_response = full_response

        # LLM parameters
        self.model_kwargs = {
            "temperature": kwargs.get("temperature", 0.01),
            "maxTokens": kwargs.get("max_tokens", 2048),
            "topP": kwargs.get("top_p", 0.9),
        }

        # AWS parameters
        self.region = kwargs.get("region", "eu-central-1")

        # Initialize the client
        self.client = boto3.client("bedrock-runtime", self.region)

    def invoke(self, prompt: str | Messages, **override_kwargs) -> str:
        """
        Call the model with the given prompt
        :param prompt: text to feed the model
        :param override_kwargs: Additional keyword arguments to override the default model kwargs
        :return: response
        """

        # Update model kwargs with any overrides
        original_model_kwargs = self.model_kwargs.copy()
        self.model_kwargs.update(override_kwargs)

        # Call the model
        response = self._invoke(prompt=prompt)

        # Get response completion
        response_body = response.get("output", {}).get("message", None)

        # Get token counts
        input_token_count = response.get("usage", {}).get("inputTokens", 0)
        output_token_count = response.get("usage", {}).get("outputTokens", 0)

        # Get completion
        if response_body and "content" in response_body and response_body["content"]:
            completion = response_body["content"][0].get("text", "")
        else:
            completion = ""

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

        # Reset model kwargs
        self.model_kwargs = original_model_kwargs

        # Depending on the full_response flag, return either the full response or just the completion
        return response if self.full_response else completion

    def _invoke(self, prompt: str | Messages) -> dict:
        """
        Handle the API request to the service
        :param prompt: text to feed the model
        :return: response completion
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

        # Build request body
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            **messages,
            **self.model_kwargs,
        }

        logger.debug(f"Request body: {request_body}")

        # Prepare the request parameters
        invoke_params = {
            "modelId": self.model_id,
            "messages": messages["messages"],
            "inferenceConfig": self.model_kwargs,
        }
        if "system" in messages:
            invoke_params["system"] = messages["system"]

        # Shooting our shot
        try:
            response = self.client.converse(**invoke_params)

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
