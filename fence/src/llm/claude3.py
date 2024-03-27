########################
# Claude gen 3 classes #
########################

import json
from datadog_lambda.metric import lambda_metric
import boto3
from fence.src.llm.base import LLM


MODEL_ID_SONNET = "anthropic.claude-3-sonnet-20240229-v1:0"
MODEL_ID_HAIKU = "anthropic.claude-3-haiku-20240307-v1:0"

class Claude3Base(LLM):
    """Base class for Claude (gen 3) models"""

    model_id = None
    llm_name = None
    inference_type = "bedrock"

    def __init__(self, source: str, full_response:bool=False, **kwargs):
        """
        Initialize a Claude model

        :param str source: An indicator of where (e.g., which feature) the model is operating from.
        :param full_response: whether to return the full response or just the TEXT completion
        :param **kwargs: Additional keyword arguments
        """

        self.source = source
        self.full_response = full_response
        self.region = kwargs.get("region", "eu-central-1")

        self.model_kwargs = {
            "temperature": kwargs.get("temperature", 0.01),
            "max_tokens": kwargs.get("max_tokens", 2048),
        }

        self.client = boto3.client("bedrock-runtime", self.region)

    @staticmethod
    def count_words_in_messages(messages: list[dict]):
        """
        Count the number of words in a list of messages. Takes all roles into account. Type must be 'text'
        :param messages: list of messages
        :return: word count (int)
        """

        word_count = 0

        # Loop through messages
        for message in messages:

            # Get content
            content = message.get("content", [])

            # Count words in content if type is 'text'
            for item in content:
                if item.get("type") == "text":
                    text = item.get("text", "")
                    word_count += len(text.split())
        return word_count

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

        metric_prefix = "showpad.data_science.llm.tokens"
        metric_name = f"{metric_prefix}.{metric_suffix}"
        tags = [
            "team:data-science",
            f"llm:{self.llm_name}",
            f"source:{self.source}",
            f"inference_type:{self.inference_type}",
        ]

        lambda_metric(metric_name=f"{metric_name}_words", value=word_count, tags=tags)
        lambda_metric(
            metric_name=f"{metric_name}_characters", value=token_count, tags=tags
        )

    def invoke(self, prompt: str | list, **kwargs) -> str:
        """
        Call the model with the given prompt
        :param prompt: text to feed the model
        :param bare_bones: whether to use the 'Human: ... Assistant: ...' formatting
        :return: response
        """

        # Call the model
        response = self._invoke(prompt=prompt, **kwargs)

        # Get response completion
        response_body = json.loads(response.get("body").read().decode())

        # Get token counts
        input_token_count = response_body['usage']['input_tokens']
        output_token_count = response_body['usage']['output_tokens']

        # Get completion
        completion = response_body["content"][0]["text"]

        # Get input and output word count
        if isinstance(prompt, str):
            input_word_count = len(prompt.split())
        elif isinstance(prompt, list):
            input_word_count = self.count_words_in_messages(prompt)
        else:
            raise ValueError("Prompt must be a string or a list of messages")
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

    def _invoke(self, prompt: str|list, **kwargs) -> dict:
        """
        Handle the API request to the service
        :param prompt: text to feed the model
        :return: response completion
        """

        # Format prompt: Claude3 models expect a list of messages, with content, roles, etc.
        # However, if we receive a string, we will format it as a single user message for ease of use.
        if isinstance(prompt, list):
            messages = prompt
        elif isinstance(prompt, str):
            messages = [
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
        else:
            raise ValueError("Prompt must be a string or a list of messages")

        # Build request body
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "messages": messages,
            **self.model_kwargs,
            **kwargs
        }

        # Send request
        try:
            response = self.client.invoke_model(
                modelId=self.model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(request_body)
            )

            # Log invocation
            self.invocation_logging()

            return response

        except Exception as e:
            raise ValueError(f"Error raised by bedrock service: {e}")


class ClaudeHaiku(Claude3Base):
    """Claude Haiku model class"""

    def __init__(self, source: str, **kwargs):
        """
        Initialize a Claude Haiku model
        :param str source: An indicator of where (e.g., which feature) the model is operating from.
        :param **kwargs: Additional keyword arguments
        """

        super().__init__(source=source, **kwargs)

        self.model_id = MODEL_ID_HAIKU
        self.llm_name = "ClaudeHaiku"

class ClaudeSonnet(Claude3Base):
    """Claude Sonnet model class"""

    def __init__(self, source: str, **kwargs):
        """
        Initialize a Claude Sonnet model
        :param str source: An indicator of where (e.g., which feature) the model is operating from.
        :param **kwargs: Additional keyword arguments
        """

        super().__init__(source=source, **kwargs)

        self.model_id = MODEL_ID_SONNET
        self.llm_name = "ClaudeSonnet"






if __name__ == '__main__':

    # Create an instance of the ClaudeHaiku class
    claude_haiku = ClaudeHaiku(source='test', region='us-east-1')

    # Call the invoke method with a prompt
    response = claude_haiku.invoke(prompt='The sun is shining brightly')

    # Create an instance of the ClaudeSonnet class
    claude_sonnet = ClaudeSonnet(source='test', region='us-east-1')

    # Call the invoke method with a prompt
    prompt = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Shall I compare thee to a summer's day?"
                }
            ]
        }
    ]
    response2 = claude_sonnet.invoke(prompt=prompt)
