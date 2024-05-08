####################
# Base LLM classes #
####################

import math
from abc import ABC, abstractmethod

from datadog_lambda.metric import lambda_metric


class LLM(ABC):
    """
    Base class for LLMs
    """

    model_name = None
    llm_name = None
    inference_type = None

    def __call__(self, prompt: str, **kwargs) -> str:
        return self.invoke(prompt, **kwargs)

    @abstractmethod
    def invoke(self, prompt: str | list) -> str:
        raise NotImplementedError

    def invocation_logging(self):
        lambda_metric(
            metric_name="showpad.data_science.llm.exam_generation.invocation",
            value=1,
            tags=[
                "team:data-science",
                f"llm:{self.llm_name}",
                f"source:{self.source}",
                f"inference_type:{self.inference_type}",
            ],
        )

    @staticmethod
    def calculate_token_metrics(text: str) -> tuple[int, int]:
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
