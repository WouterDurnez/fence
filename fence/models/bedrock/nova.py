"""
Claude Gen 3 models
"""

import logging

from fence.models.bedrock.base import BedrockBase

MODEL_ID_PRO = "amazon.nova-pro-v1:0"
MODEL_ID_LITE = "amazon.nova-lite-v1:0"
MODEL_ID_MICRO = "amazon.nova-micro-v1:0"


logger = logging.getLogger(__name__)


class NovaPro(BedrockBase):
    """Amazon Nova Pro model class"""

    def __init__(self, **kwargs):
        """
        Initialize a Claude Haiku model
        :param str source: An indicator of where (e.g., which feature) the model is operating from.
        :param **kwargs: Additional keyword arguments
        """

        super().__init__(**kwargs)

        self.model_id = MODEL_ID_PRO
        self.model_name = "Nova Pro"


class NovaLite(BedrockBase):
    """Amazon Nova Lite model class"""

    def __init__(self, **kwargs):
        """
        Initialize a Claude Haiku model
        :param str source: An indicator of where (e.g., which feature) the model is operating from.
        :param **kwargs: Additional keyword arguments
        """

        super().__init__(**kwargs)

        self.model_id = MODEL_ID_LITE
        self.model_name = "Nova Lite"


class NovaMicro(BedrockBase):
    """Amazon Nova Micro model class"""

    def __init__(self, **kwargs):
        """
        Initialize a Claude Haiku model
        :param str source: An indicator of where (e.g., which feature) the model is operating from.
        :param **kwargs: Additional keyword arguments
        """

        super().__init__(**kwargs)

        self.model_id = MODEL_ID_MICRO
        self.model_name = "Nova Micro"


if __name__ == "__main__":

    # Create model with tools
    nova_with_tools = NovaPro(
        source="test",
        metric_prefix="supertest",
        extra_tags={"test": "test"},
        region="us-east-1",
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
    nova_without_tools = NovaPro(
        source="test",
        metric_prefix="supertest",
        extra_tags={"test": "test"},
        region="us-east-1",
    )

    prompt = "What is the top music played on WABC?"

    # Test with tools
    print("\n=== Testing with tools ===")

    print("\n1. Invoke without full_response:")
    response = nova_with_tools.invoke(prompt)
    print(response)

    print("\n2. Invoke with full_response:")
    response = nova_with_tools.invoke(prompt, full_response=True)
    print(response)

    print("\n3. Stream without full_response:")
    for chunk in nova_with_tools.stream(prompt):
        print(chunk, end="")

    print("\n4. Stream with full_response:")
    for chunk in nova_with_tools.stream(prompt, full_response=True):
        print(chunk)

    # Test without tools
    print("\n=== Testing without tools ===")

    print("\n1. Invoke without full_response:")
    response = nova_without_tools.invoke(prompt)
    print(response)

    print("\n2. Invoke with full_response:")
    response = nova_without_tools.invoke(prompt, full_response=True)
    print(response)

    print("\n3. Stream without full_response:")
    for chunk in nova_without_tools.stream(prompt):
        print(chunk, end="")

    print("\n4. Stream with full_response:")
    for chunk in nova_without_tools.stream(prompt, full_response=True):
        print(chunk)
