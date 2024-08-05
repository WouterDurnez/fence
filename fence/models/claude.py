"""
Claude Gen 1/2 models
"""

import json

import boto3

from fence.models.base import (
    LLM,
    get_log_callback,
    get_log_tags,
    register_log_callback,
    register_log_tags,
)


class ClaudeBase(LLM):
    """Base class for Claude (gen 1/2) models"""

    model_name: str | None = None
    llm_name: str | None = None
    inference_type = "bedrock"
    logging_tags: dict = {}

    def __init__(
        self,
        source: str,
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

        self.source = source

        # LLM parameters
        self.temperature = kwargs.get("temperature", 0.01)
        self.max_tokens_to_sample = kwargs.get("max_tokens_to_sample", 2048)
        self.model_kwargs = {
            "temperature": self.temperature,
            "max_tokens_to_sample": self.max_tokens_to_sample,
        }

        # AWS parameters
        self.region = kwargs.get("region", "eu-central-1")
        self.client = boto3.client("bedrock-runtime", self.region)

        # Logging parameters
        self.metric_prefix = metric_prefix or ""
        self.logging_tags = get_log_tags() or {}
        self.logging_tags.update(extra_tags or {})

    # def send_token_metrics(
    #     self, metric_suffix: str, token_count: int, word_count: int
    # ) -> None:
    #     """
    #     Send word and character token metrics to Datadog
    #     :param metric_suffix: metric suffix to concatenate to the metric name
    #     :param token_count: token count
    #     :param word_count: word count
    #     :return:
    #     """
    #
    #     # Note: keeping this here because other models may include token metrics
    #     # in their response, so we don't need to calculate them again.
    #
    #     metric_prefix = "showpad.data_science.llm.tokens"
    #     metric_name = f"{metric_prefix}.{metric_suffix}"
    #     tags = [
    #         "team:data-science",
    #         f"llm:{self.llm_name}",
    #         f"source:{self.source}",
    #         f"inference_type:{self.inference_type}",
    #     ]
    #
    #     lambda_metric(metric_name=f"{metric_name}_words", value=word_count, tags=tags)
    #     lambda_metric(
    #         metric_name=f"{metric_name}_characters", value=token_count, tags=tags
    #     )

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

            log_callback(
                # Add metrics
                {
                    f"{self.metric_prefix}.{self.source}.invocation": 1,
                    f"{self.metric_prefix}.{self.source}.input_token_count": input_token_count,
                    f"{self.metric_prefix}.{self.source}.output_token_count": output_token_count,
                    f"{self.metric_prefix}.{self.source}.input_word_count": input_word_count,
                    f"{self.metric_prefix}.{self.source}.output_word_count": output_word_count,
                },
                # Format tags as ['key:value', 'key:value', ...]
                [f"{k}:{v}" for k, v in self.logging_tags.items()],
            )

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
                modelId=self.model_name,
                accept=accept,
                contentType=contentType,
            )

            return response

        except Exception as e:
            raise ValueError(f"Error raised by bedrock service: {e}")


class ClaudeInstant(ClaudeBase):
    """Claude Instant model class"""

    def __init__(self, source: str, **kwargs):
        """
        Initialize a Claude Instant model
        :param str source: An indicator of where (e.g., which feature) the model is operating from.
        :param **kwargs: Additional keyword arguments
        """

        super().__init__(source=source, **kwargs)

        self.model_name = "anthropic.claude-instant-v1"
        self.llm_name = "ClaudeInstant"


class ClaudeV2(ClaudeBase):
    """Claude v2 model class"""

    def __init__(self, source: str, **kwargs):
        """
        Initialize a Claude Instant model
        :param str source: An indicator of where (e.g., which feature) the model is operating from.
        :param **kwargs: Additional keyword arguments
        """

        super().__init__(source=source, **kwargs)

        self.model_name = "anthropic.anthropic.claude-v2"
        self.llm_name = "ClaudeV2"


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
