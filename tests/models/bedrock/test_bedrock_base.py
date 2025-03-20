import pytest

from fence.models.base import Messages
from fence.models.bedrock.base import BedrockBase


# Create a mock model class that inherits from BedrockBase
class MockBedrockModel(BedrockBase):
    """Mock Bedrock model for testing"""

    def __init__(self, **kwargs):
        """
        Initialize mock model with a predefined model_id
        :param kwargs: Additional keyword arguments
        """
        super().__init__(**kwargs)
        self.model_id = "MODEL_ID_HAIKU"
        self.inferenceConfig = {
            "temperature": 1,
            "max_tokens": None,
        }
        self.toolConfig = kwargs.get(
            "toolConfig",
            {
                "stream": False,
            },
        )

    def _invoke(self, prompt: str | Messages, stream=False, **kwargs) -> str | dict:
        """
        Override _invoke to simulate a successful response without actual AWS call
        :param prompt: The input prompt
        :param stream: Whether to stream the response
        :param kwargs: Additional keyword arguments
        :return: The mock response text or full response object
        """
        self._check_if_prompt_is_valid(prompt)

        # Simulate tool call response if toolConfig is present
        if self.toolConfig and "tools" in self.toolConfig:
            if self.full_response:
                if stream:
                    return [
                        {
                            "messageStart": {"role": "assistant"},
                        },
                        {
                            "contentBlockDelta": {
                                "delta": {
                                    "toolCall": {
                                        "name": "top_song",
                                        "arguments": {"sign": "WABC"},
                                    }
                                }
                            }
                        },
                        {"messageStop": {"stopReason": "tool_call"}},
                    ]
                return {
                    "output": {
                        "message": {
                            "content": [
                                {
                                    "toolCall": {
                                        "name": "top_song",
                                        "arguments": {"sign": "WABC"},
                                    }
                                }
                            ]
                        }
                    },
                    "usage": {"inputTokens": 10, "outputTokens": 20},
                }
            if stream:
                return ["I'll check the top song for WABC."]
            return "I'll check the top song for WABC."

        # Regular text response
        if self.full_response:
            if stream:
                return [
                    {
                        "messageStart": {"role": "assistant"},
                    },
                    {
                        "contentBlockDelta": {"delta": {"text": "Test response"}},
                    },
                    {"messageStop": {"stopReason": "end_turn"}},
                ]
            return {
                "output": {"message": {"content": [{"text": "Test response"}]}},
                "usage": {"inputTokens": 10, "outputTokens": 20},
            }
        if stream:
            return ["Test response"]
        return "Test response"


class MockBedrockModelNoConfig(BedrockBase):
    """Mock Bedrock model for testing without configs"""

    def __init__(self, **kwargs):
        """
        Initialize mock model with a predefined model_id and no configs
        :param kwargs: Additional keyword arguments
        """
        super().__init__(**kwargs)
        self.model_id = "MODEL_ID_HAIKU"


class TestBedrockBase:
    @pytest.fixture
    def bedrock_base(self):
        """Create a MockBedrockModel instance for testing"""
        return MockBedrockModel(
            source="test_source",
            full_response=False,
            metric_prefix="test_prefix",
            extra_tags={"test_tag": "value"},
        )

    @pytest.fixture
    def bedrock_base_full_response(self):
        """Create a MockBedrockModel instance with full_response=True for testing"""
        return MockBedrockModel(
            source="test_source",
            full_response=True,
            metric_prefix="test_prefix",
            extra_tags={"test_tag": "value"},
        )

    @pytest.fixture
    def bedrock_base_no_config(self):
        """Create a MockBedrockModelNoConfig instance for testing"""
        return MockBedrockModelNoConfig(
            source="test_source",
            full_response=False,
            metric_prefix="test_prefix",
            extra_tags={"test_tag": "value"},
        )

    @pytest.fixture
    def bedrock_base_with_tools(self):
        """Create a MockBedrockModel instance with tools for testing"""
        return MockBedrockModel(
            source="test_source",
            full_response=False,
            metric_prefix="test_prefix",
            extra_tags={"test_tag": "value"},
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
                                            "description": "The call sign for the radio station",
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

    @pytest.fixture
    def bedrock_base_with_tools_full_response(self):
        """Create a MockBedrockModel instance with tools and full_response=True for testing"""
        return MockBedrockModel(
            source="test_source",
            full_response=True,
            metric_prefix="test_prefix",
            extra_tags={"test_tag": "value"},
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
                                            "description": "The call sign for the radio station",
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

    def test_initialization(self, bedrock_base):
        """Test that the BedrockBase is initialized correctly"""
        assert bedrock_base.source == "test_source"
        assert bedrock_base.metric_prefix == "test_prefix"
        assert bedrock_base.logging_tags == {"test_tag": "value"}
        assert bedrock_base.model_id == "MODEL_ID_HAIKU"

        # Check default config values
        assert bedrock_base.inferenceConfig == {
            "temperature": 1,
            "max_tokens": None,
        }
        assert bedrock_base.toolConfig == {
            "stream": False,
        }

    def test_initialization_without_configs(self, bedrock_base_no_config):
        """Test that the BedrockBase is initialized correctly without configs"""
        assert bedrock_base_no_config.source == "test_source"
        assert bedrock_base_no_config.metric_prefix == "test_prefix"
        assert bedrock_base_no_config.logging_tags == {"test_tag": "value"}
        assert bedrock_base_no_config.model_id == "MODEL_ID_HAIKU"

        # Check that configs are None when not provided
        assert bedrock_base_no_config.inferenceConfig is None
        assert bedrock_base_no_config.toolConfig is None

    def test_invoke_with_different_prompt_types(self, bedrock_base):
        """Test invoking with different prompt types"""
        # Test with string prompt
        str_response = bedrock_base.invoke("Test prompt")
        assert str_response == "Test response"

        # Test with Messages prompt
        messages = Messages(messages=[{"role": "user", "content": "Test prompt"}])
        msg_response = bedrock_base.invoke(messages)
        assert msg_response == "Test response"

    def test_invoke_with_override_kwargs(self, bedrock_base):
        """Test invoking with override kwargs"""
        response = bedrock_base.invoke("Test prompt", temperature=0.7, maxTokens=1024)
        assert response == "Test response"

    def test_invalid_prompt_type(self, bedrock_base):
        """Test invoking with an invalid prompt type"""
        with pytest.raises(
            ValueError, match="Prompt must be a string or a Messages object!"
        ):
            print(bedrock_base.invoke(123))  # Invalid type

    def test_invoke_with_full_response(self, bedrock_base_full_response):
        """Test invoking with full_response=True"""
        response = bedrock_base_full_response.invoke("Test prompt")
        assert isinstance(response, dict)
        assert response["output"]["message"]["content"][0]["text"] == "Test response"
        assert response["usage"]["inputTokens"] == 10
        assert response["usage"]["outputTokens"] == 20

    def test_invoke_with_tools(self, bedrock_base_with_tools):
        """Test invoking with tools"""
        response = bedrock_base_with_tools.invoke("Test prompt")
        assert response == "I'll check the top song for WABC."

    def test_invoke_with_tools_full_response(
        self, bedrock_base_with_tools_full_response
    ):
        """Test invoking with tools and full_response=True"""
        response = bedrock_base_with_tools_full_response.invoke("Test prompt")
        assert isinstance(response, dict)
        assert (
            response["output"]["message"]["content"][0]["toolCall"]["name"]
            == "top_song"
        )
        assert (
            response["output"]["message"]["content"][0]["toolCall"]["arguments"]["sign"]
            == "WABC"
        )
        assert response["usage"]["inputTokens"] == 10
        assert response["usage"]["outputTokens"] == 20

    def test_stream_with_tools(self, bedrock_base_with_tools):
        """Test streaming with tools"""
        chunks = list(bedrock_base_with_tools.stream("Test prompt"))
        assert len(chunks) == 1
        assert chunks[0] == "I'll check the top song for WABC."

    def test_stream_with_tools_full_response(
        self, bedrock_base_with_tools_full_response
    ):
        """Test streaming with tools and full_response=True"""
        chunks = list(bedrock_base_with_tools_full_response.stream("Test prompt"))
        assert len(chunks) == 3  # messageStart, contentBlockDelta, messageStop
        assert chunks[0]["messageStart"]["role"] == "assistant"
        assert chunks[1]["contentBlockDelta"]["delta"]["toolCall"]["name"] == "top_song"
        assert chunks[2]["messageStop"]["stopReason"] == "tool_call"

    def test_stream_without_tools(self, bedrock_base):
        """Test streaming without tools"""
        chunks = list(bedrock_base.stream("Test prompt"))
        assert len(chunks) == 1
        assert chunks[0] == "Test response"

    def test_stream_without_tools_full_response(self, bedrock_base_full_response):
        """Test streaming without tools and full_response=True"""
        chunks = list(bedrock_base_full_response.stream("Test prompt"))
        assert len(chunks) == 3  # messageStart, contentBlockDelta, messageStop
        assert chunks[0]["messageStart"]["role"] == "assistant"
        assert chunks[1]["contentBlockDelta"]["delta"]["text"] == "Test response"
        assert chunks[2]["messageStop"]["stopReason"] == "end_turn"
