"""
Amazon Nova models
"""

import logging
from typing import Literal

from fence.models.bedrock.base import BedrockBase
from fence.utils.logger import setup_logging

# Default model IDs (without context window suffix)
MODEL_ID_PRO = "amazon.nova-pro-v1:0"
MODEL_ID_LITE = "amazon.nova-lite-v1:0"
MODEL_ID_MICRO = "amazon.nova-micro-v1:0"
MODEL_ID_LITE_2 = "amazon.nova-2-lite-v1:0"

# Valid context window options for each model
NOVA_PRO_CONTEXT_WINDOWS = ["24k", "300k"]
NOVA_LITE_CONTEXT_WINDOWS = ["24k", "300k"]
NOVA_MICRO_CONTEXT_WINDOWS = ["24k", "128k"]
NOVA_LITE_2_CONTEXT_WINDOWS = ["256k"]

logger = logging.getLogger(__name__)


class NovaPro(BedrockBase):
    """Amazon Nova Pro model class"""

    def __init__(self, context_window: Literal["24k", "300k"] | None = None, **kwargs):
        """
        Initialize a Nova Pro model
        :param context_window: Context window size. Valid options: "24k", "300k". Defaults to None (uses base model).
        :param str source: An indicator of where (e.g., which feature) the model is operating from.
        :param **kwargs: Additional keyword arguments
        """
        if context_window is not None and context_window not in NOVA_PRO_CONTEXT_WINDOWS:
            raise ValueError(
                f"Invalid context_window '{context_window}' for NovaPro. "
                f"Valid options: {NOVA_PRO_CONTEXT_WINDOWS}"
            )

        self.model_id = MODEL_ID_PRO
        if context_window:
            self.model_id = f"{self.model_id}:{context_window}"

        self.model_name = "Nova Pro"
        if context_window:
            self.model_name = f"Nova Pro ({context_window})"

        super().__init__(**kwargs)


class NovaLite(BedrockBase):
    """Amazon Nova Lite model class"""

    def __init__(self, context_window: Literal["24k", "300k"] | None = None, **kwargs):
        """
        Initialize a Nova Lite model
        :param context_window: Context window size. Valid options: "24k", "300k". Defaults to None (uses base model).
        :param str source: An indicator of where (e.g., which feature) the model is operating from.
        :param **kwargs: Additional keyword arguments
        """
        if context_window is not None and context_window not in NOVA_LITE_CONTEXT_WINDOWS:
            raise ValueError(
                f"Invalid context_window '{context_window}' for NovaLite. "
                f"Valid options: {NOVA_LITE_CONTEXT_WINDOWS}"
            )

        self.model_id = MODEL_ID_LITE
        if context_window:
            self.model_id = f"{self.model_id}:{context_window}"

        self.model_name = "Nova Lite"
        if context_window:
            self.model_name = f"Nova Lite ({context_window})"

        super().__init__(**kwargs)


class NovaMicro(BedrockBase):
    """Amazon Nova Micro model class"""

    def __init__(self, context_window: Literal["24k", "128k"] | None = None, **kwargs):
        """
        Initialize a Nova Micro model
        :param context_window: Context window size. Valid options: "24k", "128k". Defaults to None (uses base model).
        :param str source: An indicator of where (e.g., which feature) the model is operating from.
        :param **kwargs: Additional keyword arguments
        """
        if context_window is not None and context_window not in NOVA_MICRO_CONTEXT_WINDOWS:
            raise ValueError(
                f"Invalid context_window '{context_window}' for NovaMicro. "
                f"Valid options: {NOVA_MICRO_CONTEXT_WINDOWS}"
            )

        self.model_id = MODEL_ID_MICRO
        if context_window:
            self.model_id = f"{self.model_id}:{context_window}"

        self.model_name = "Nova Micro"
        if context_window:
            self.model_name = f"Nova Micro ({context_window})"

        super().__init__(**kwargs)


class Nova2Lite(BedrockBase):
    """Amazon Nova 2 Lite model class"""

    def __init__(self, context_window: Literal["256k"] | None = None, **kwargs):
        """
        Initialize a Nova 2 Lite model
        :param context_window: Context window size. Valid options: "256k". Defaults to None (uses base model).
        :param str source: An indicator of where (e.g., which feature) the model is operating from.
        :param **kwargs: Additional keyword arguments
        """
        if context_window is not None and context_window not in NOVA_LITE_2_CONTEXT_WINDOWS:
            raise ValueError(
                f"Invalid context_window '{context_window}' for Nova2Lite. "
                f"Valid options: {NOVA_LITE_2_CONTEXT_WINDOWS}"
            )

        self.model_id = MODEL_ID_LITE_2
        if context_window:
            self.model_id = f"{self.model_id}:{context_window}"

        self.model_name = "Nova 2 Lite"
        if context_window:
            self.model_name = f"Nova 2 Lite ({context_window})"

        super().__init__(**kwargs)


class Nova2Lite256K(BedrockBase):
    """
    Amazon Nova 2 Lite 256K model class (deprecated).

    Use Nova2Lite(context_window="256k") instead for better flexibility.
    This class is kept for backward compatibility.
    """

    def __init__(self, **kwargs):
        """
        Initialize a Nova 2 Lite 256K model
        :param str source: An indicator of where (e.g., which feature) the model is operating from.
        :param **kwargs: Additional keyword arguments
        """
        self.model_id = MODEL_ID_LITE_2
        self.model_id = f"{self.model_id}:256k"
        self.model_name = "Nova 2 Lite (256k)"

        super().__init__(**kwargs)



if __name__ == "__main__":

    setup_logging(log_level="DEBUG")

    # Create model with tools
    nova_with_tools = NovaPro(
        source="test",
        cross_region="eu",
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
    for chunk in nova_without_tools.stream(prompt, full_response=False):
        print(chunk, end="")
