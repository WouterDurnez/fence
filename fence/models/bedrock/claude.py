"""
Claude models
"""

import logging

from fence.models.base import register_log_callback, register_log_tags
from fence.models.bedrock.base import BedrockBase

logger = logging.getLogger(__name__)

MODEL_ID_V2 = "anthropic.anthropic.claude-v2"
MODEL_ID_INSTANT = "anthropic.claude-instant-v1"
MODEL_ID_SONNET = "anthropic.claude-3-sonnet-20240229-v1:0"
MODEL_ID_SONNET_3_5_V2 = "anthropic.claude-3-5-sonnet-20241022-v2:0"
MODEL_ID_SONNET_3_5 = "anthropic.claude-3-5-sonnet-20240620-v1:0"
MODEL_ID_HAIKU = "anthropic.claude-3-haiku-20240307-v1:0"


class ClaudeInstant(BedrockBase):
    """Claude Instant model class"""

    def __init__(self, **kwargs):
        """
        Initialize a Claude Instant model
        :param str source: An indicator of where (e.g., which feature) the model is operating from.
        :param **kwargs: Additional keyword arguments
        """

        super().__init__(**kwargs)

        self.model_id = MODEL_ID_INSTANT
        self.model_name = "Claude Instant"


class ClaudeHaiku(BedrockBase):
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


class ClaudeSonnet(BedrockBase):
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


class Claude35Sonnet(BedrockBase):
    """Claude 3.5 Sonnet V1 model class"""

    def __init__(self, **kwargs):
        """
        Initialize a Claude Sonnet 3.5 model
        :param str source: An indicator of where (e.g., which feature) the model is operating from.
        :param **kwargs: Additional keyword arguments
        """

        super().__init__(**kwargs)

        self.model_id = MODEL_ID_SONNET_3_5
        self.model_name = "Claude 3.5 Sonnet"


class Claude35SonnetV2(BedrockBase):
    """Claude 3.5 Sonnet V2 model class"""

    def __init__(self, **kwargs):
        """
        Initialize a Claude Sonnet 3 model
        :param str source: An indicator of where (e.g., which feature) the model is operating from.
        :param **kwargs: Additional keyword arguments
        """

        super().__init__(**kwargs)

        self.model_id = MODEL_ID_SONNET_3_5_V2
        self.model_name = "Claude 3 Sonnet"


if __name__ == "__main__":

    # Register logging callback
    register_log_callback(lambda metrics, tags: print(metrics, tags))

    # Register logging tags
    register_log_tags({"team": "data-science-test", "project": "fence"})

    # Initialize and test models
    for model in [
        ClaudeInstant,
        ClaudeHaiku,
        ClaudeSonnet,
        Claude35Sonnet,
        Claude35SonnetV2,
    ]:

        print(f"Testing {model.__name__}...")

        # Create an instance of the model class
        claude = model(
            source="test",
            metric_prefix="supertest",
            extra_tags={"test": "test"},
            region="us-east-1",
        )

        # Call the model
        response = claude("Hello, who are you?")

        # Print the response
        print(response)
