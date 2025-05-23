"""
Claude models
"""

import logging

from fence.models.bedrock.base import BedrockBase

logger = logging.getLogger(__name__)

MODEL_ID_V2 = "anthropic.anthropic.claude-v2"
MODEL_ID_INSTANT = "anthropic.claude-instant-v1"
MODEL_ID_SONNET = "anthropic.claude-3-sonnet-20240229-v1:0"
MODEL_ID_SONNET_3_5_V2 = "anthropic.claude-3-5-sonnet-20241022-v2:0"
MODEL_ID_SONNET_3_5 = "anthropic.claude-3-5-sonnet-20240620-v1:0"
MODEL_ID_SONNET_3_7 = "anthropic.claude-3-7-sonnet-20250219-v1:0"
MODEL_ID_HAIKU = "anthropic.claude-3-haiku-20240307-v1:0"
MODEL_ID_SONNET_4 = "anthropic.claude-sonnet-4-20250514-v1:0"
MODEL_ID_OPUS_4 = "anthropic.claude-opus-4-20250514-v1:0"


class ClaudeInstant(BedrockBase):
    """Claude Instant model class"""

    def __init__(self, **kwargs):
        """
        Initialize a Claude Instant model
        :param str source: An indicator of where (e.g., which feature) the model is operating from.
        :param **kwargs: Additional keyword arguments
        """

        self.model_id = MODEL_ID_INSTANT
        self.model_name = "Claude Instant"

        super().__init__(**kwargs)


class ClaudeHaiku(BedrockBase):
    """Claude Haiku model class"""

    def __init__(self, **kwargs):
        """
        Initialize a Claude Haiku model
        :param str source: An indicator of where (e.g., which feature) the model is operating from.
        :param **kwargs: Additional keyword arguments
        """

        self.model_id = MODEL_ID_HAIKU
        self.model_name = "Claude Haiku"

        super().__init__(**kwargs)


class ClaudeSonnet(BedrockBase):
    """Claude Sonnet model class"""

    def __init__(self, **kwargs):
        """
        Initialize a Claude Sonnet model
        :param str source: An indicator of where (e.g., which feature) the model is operating from.
        :param **kwargs: Additional keyword arguments
        """

        self.model_id = MODEL_ID_SONNET
        self.model_name = "Claude Sonnet"

        super().__init__(**kwargs)


class Claude35Sonnet(BedrockBase):
    """Claude 3.5 Sonnet V1 model class"""

    def __init__(self, **kwargs):
        """
        Initialize a Claude Sonnet 3.5 model
        :param str source: An indicator of where (e.g., which feature) the model is operating from.
        :param **kwargs: Additional keyword arguments
        """

        self.model_id = MODEL_ID_SONNET_3_5
        self.model_name = "Claude 3.5 Sonnet"

        super().__init__(**kwargs)


class Claude35SonnetV2(BedrockBase):
    """Claude 3.5 Sonnet V2 model class"""

    def __init__(self, **kwargs):
        """
        Initialize a Claude Sonnet 3 model
        :param str source: An indicator of where (e.g., which feature) the model is operating from.
        :param **kwargs: Additional keyword arguments
        """

        self.model_id = MODEL_ID_SONNET_3_5_V2
        self.model_name = "Claude 3.5 Sonnet V2"

        super().__init__(**kwargs)


class Claude37Sonnet(BedrockBase):
    """Claude 3.7 Sonnet model class"""

    def __init__(self, **kwargs):
        """
        Initialize a Claude 3.7 Sonnet model
        :param str source: An indicator of where (e.g., which feature) the model is operating from.
        :param **kwargs: Additional keyword arguments
        """

        self.model_id = MODEL_ID_SONNET_3_7
        self.model_name = "Claude 3.7 Sonnet"

        super().__init__(**kwargs)


class Claude4Sonnet(BedrockBase):
    """Claude 4 Sonnet model class"""

    def __init__(self, **kwargs):
        """
        Initialize a Claude 4 Sonnet model
        :param str source: An indicator of where (e.g., which feature) the model is operating from.
        :param **kwargs: Additional keyword arguments
        """

        self.model_id = MODEL_ID_SONNET_4
        self.model_name = "Claude 4 Sonnet"

        super().__init__(**kwargs)


class Claude4Opus(BedrockBase):
    """Claude 4 Opus model class"""

    def __init__(self, **kwargs):
        """
        Initialize a Claude 4 Opus model
        :param str source: An indicator of where (e.g., which feature) the model is operating from.
        :param **kwargs: Additional keyword arguments
        """

        self.model_id = MODEL_ID_OPUS_4
        self.model_name = "Claude 4 Opus"

        super().__init__(**kwargs)


if __name__ == "__main__":

    # Create model with tools
    claude_with_tools = Claude37Sonnet(
        cross_region="eu",
        source="test",
        metric_prefix="supertest",
        extra_tags={"test": "test"},
        region="eu-central-1",
        toolConfig={
            "tools": [
                {
                    "toolSpec": {
                        "name": "top_song",
                        "description": "Get the most popular song played on a radio station.",
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {
                                    "sign": {
                                        "type": "string",
                                        "description": "The call sign for the radio station for which you want the most popular song. Example calls signs are WZPZ, and WKRP.",
                                    }
                                },
                                "required": ["sign"],
                            }
                        },
                    }
                }
            ]
        },
    )

    # Create model without tools
    claude_without_tools = Claude35Sonnet(
        source="test",
        metric_prefix="supertest",
        extra_tags={"test": "test"},
        region="eu-central-1",
    )

    prompt = "What is the top music played on WABC?"

    # # Test with tools
    # print("\n=== Testing with tools ===")

    # print("\n1. Invoke without full_response:")
    # response = claude_with_tools.invoke(prompt)
    # print(response)

    # print("\n2. Invoke with full_response:")
    # response = claude_with_tools.invoke(prompt, full_response=True)
    # print(response)

    # print("\n3. Stream without full_response:")
    # for chunk in claude_with_tools.stream(prompt):
    #     print(chunk, end="")

    print("\n4. Stream with full_response:")
    for chunk in claude_with_tools.stream(prompt, full_response=False):
        print(chunk, end="")

    # # Test without tools
    # print("\n=== Testing without tools ===")

    # print("\n1. Invoke without full_response:")
    # response = claude_without_tools.invoke(prompt)
    # print(response)

    # print("\n2. Invoke with full_response:")
    # response = claude_without_tools.invoke(prompt, full_response=True)
    # print(response)

    # print("\n3. Stream without full_response:")
    # for chunk in claude_without_tools.stream(prompt):
    #     print(chunk, end="")

    # print("\n4. Stream with full_response:")
    # for chunk in claude_without_tools.stream(prompt, full_response=True):
    #     print(chunk)
