#################
# Model classes #
#################

import json
import math
from abc import ABC, abstractmethod
from typing import Tuple

import boto3
from datadog_lambda.metric import lambda_metric

from fence.src.utils.base import time_it


class LLM(ABC):
    def __call__(self, prompt: str, **kwargs) -> str:
        return self.invoke(prompt, **kwargs)

    @abstractmethod
    def invoke(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError


class ClaudeInstantLLM(LLM):
    """Claude Instant model class"""

    model_name = "anthropic.claude-instant-v1"
    llm_name = "ClaudeInstant"
    inference_type = "bedrock"

    def __init__(self, source: str, **kwargs):
        """
        Initialize a Claude Instant model

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

    @staticmethod
    def calculate_token_metrics(text: str) -> Tuple[int, int]:
        words = text.split()

        # Link to OpenAI helpcenter article, on how to calculate tokens
        # https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them

        # 1 token ≈ ¾ words
        num_words = len(words)
        token_words = math.ceil(num_words / 0.75)

        # 1 token ≈ 4 chars in English
        num_characters = sum(len(word) for word in words)
        token_characters = math.ceil(num_characters / 4)

        return token_words, token_characters

    def send_token_metrics(self, metric_suffix: str, text: str) -> None:
        """
        Send word and character token metrics to Datadog
        :param metric_suffix: metric suffix to concatenate to the metric name
        :param token_words: number of token words
        :param token_characters: number of token characters
        :return:
        """

        # Calculate token metrics
        token_words, token_characters = ClaudeInstantLLM.calculate_token_metrics(text)

        metric_prefix = "showpad.data_science.llm.tokens"
        metric_name = f"{metric_prefix}.{metric_suffix}"
        tags = [
            "team:data-science",
            f"llm:{ClaudeInstantLLM.llm_name}",
            f"source:{self.source}",
            f"inference_type:{ClaudeInstantLLM.inference_type}",
        ]

        lambda_metric(metric_name=f"{metric_name}_words", value=token_words, tags=tags)
        lambda_metric(
            metric_name=f"{metric_name}_characters", value=token_characters, tags=tags
        )

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

        # Calculate token metrics for the response and send them to Datadog
        self.send_token_metrics(metric_suffix="input", text=prompt)
        self.send_token_metrics(metric_suffix="output", text=response)

        return response

    @time_it(threshold=30, only_warn=True)
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
                modelId=ClaudeInstantLLM.model_name,
                accept=accept,
                contentType=contentType,
            )

            response_body = json.loads(response.get("body").read().decode())
            return response_body.get("completion")

        except Exception as e:
            raise ValueError(f"Error raised by bedrock service: {e}")
