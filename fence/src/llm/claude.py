##########################
# Claude gen 1/2 classes #
##########################

import json

import boto3
from base import LLM


class ClaudeBase(LLM):
    """Base class for Claude (gen 1/2) models"""

    model_name = None
    llm_name = None
    inference_type = "bedrock"

    def __init__(self, source: str, **kwargs):
        """
        Initialize a Claude model

        :param str source: An indicator of where (e.g., which feature) the model is operating from.
        :param **kwargs: Additional keyword arguments
        """

        self.source = source

        self.temperature = kwargs.get("temperature", 0.01)
        self.max_tokens_to_sample = kwargs.get("max_tokens_to_sample", 2048)
        self.region = kwargs.get("region", "eu-central-1")

        self.model_kwargs = {
            "temperature": self.temperature,
            "max_tokens_to_sample": self.max_tokens_to_sample,
        }

        self.client = boto3.client("bedrock-runtime", self.region)

    def send_token_metrics(
        self, metric_suffix: str, token_count: int, word_count: int
    ) -> None:
        """
        Send word and character token metrics to Datadog
        :param metric_suffix: metric suffix to concatenate to the metric name
        :param token_count: token count
        :param word_count: word count
        :return:
        """

        # Note: keeping this here because other models may include token metrics
        # in their response, so we don't need to calculate them again.

        metric_prefix = "showpad.data_science.llm.tokens"
        metric_name = f"{metric_prefix}.{metric_suffix}"
        tags = [
            "team:data-science",
            f"llm:{self.llm_name}",
            f"source:{self.source}",
            f"inference_type:{self.inference_type}",
        ]

        # lambda_metric(metric_name=f"{metric_name}_words", value=word_count, tags=tags)
        # lambda_metric(
        #     metric_name=f"{metric_name}_characters", value=token_count, tags=tags
        # )

    def invoke(self, prompt: str, bare_bones: str = False) -> str:
        """
        Call the model with the given prompt
        :param prompt: text to feed the model
        :param bare_bones: whether to use the 'Human: ... Assistant: ...' formatting
        :return: response
        """

        # Format response
        claude_prompt = prompt if bare_bones else f"\n\nHuman: {prompt}\n\nAssistant: "

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

        # Calculate token metrics for the response and send them to Datadog
        self.send_token_metrics(
            metric_suffix="input",
            token_count=input_token_count,
            word_count=input_word_count,
        )
        self.send_token_metrics(
            metric_suffix="output",
            token_count=output_token_count,
            word_count=output_word_count,
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

            # Log invocation
            self.invocation_logging()

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


if __name__ == '__main__':

    # Initialize Claude Instant model
    claude = ClaudeInstant(source="test")

    # Call the model
    response = claude("Hello, how are you?")
