"""
Tests for the BedrockBase class and related functionality.

This module contains tests for the base Bedrock model implementation, including:
1. Initialization with different configurations
2. Handling of tool configurations in various formats
3. Basic invocation patterns (sync and streaming)
4. Response handling with and without tools
5. Full response object handling
"""

import pytest

from fence.models.base import Messages
from fence.models.bedrock.base import (
    BedrockBase,
    BedrockInferenceConfig,
    BedrockJSONSchema,
    BedrockTool,
    BedrockToolConfig,
    BedrockToolInputSchema,
    BedrockToolSpec,
)

# Check if AWS credentials are available via profile
try:
    import boto3
    from botocore.exceptions import SSOTokenLoadError

    session = boto3.Session()
    credentials = session.get_credentials()
    if credentials:
        try:
            # Try to get a token to verify credentials are valid
            sts = session.client("sts")
            sts.get_caller_identity()
            has_aws_credentials = True
        except SSOTokenLoadError:
            has_aws_credentials = False
        except Exception:
            has_aws_credentials = False
    else:
        has_aws_credentials = False
except (ImportError, Exception) as e:
    has_aws_credentials = False
    print(f"AWS credentials check failed: {str(e)}")


# Create a mock model class that inherits from BedrockBase
class MockBedrockModel(BedrockBase):
    """
    Mock implementation of the BedrockBase class for testing purposes.

    This class overrides the _invoke method to simulate API responses without
    making actual calls to the Bedrock service. It handles different response
    formats based on configuration (tools, streaming, full_response).
    """

    def __init__(self, **kwargs):
        """
        Initialize mock model with a predefined model_id.

        :param kwargs: Additional keyword arguments to pass to the parent class
        """
        super().__init__(**kwargs)
        self.model_id = "MODEL_ID_HAIKU"

    def _invoke(
        self, prompt: str | Messages, stream=False, **kwargs
    ) -> str | dict | list:
        """
        Simulate model invocation with appropriate mock responses.

        This method returns different responses based on:
        - Whether toolConfig is present and has tools
        - Whether full_response is True/False
        - Whether streaming is enabled

        :param prompt: The input prompt
        :param stream: Whether to stream the response
        :param kwargs: Additional keyword arguments
        :return: Mock response mimicking the Bedrock API response format
        """
        self._check_if_prompt_is_valid(prompt)

        # Check if toolConfig exists and has tools
        has_tools = False
        if isinstance(self.toolConfig, BedrockToolConfig) and self.toolConfig.tools:
            has_tools = True
        elif isinstance(self.toolConfig, dict) and "tools" in self.toolConfig:
            has_tools = True

        # Simulate tool call response if tools are present
        if has_tools:
            return self._generate_tool_response(stream)

        # Generate regular text response
        return self._generate_text_response(stream)

    def _generate_tool_response(self, stream: bool) -> str | dict | list:
        """
        Generate a mock response containing a tool call.

        :param stream: Whether to stream the response
        :return: Tool response in the appropriate format
        """
        if self.full_response:
            if stream:
                # Streamed full response with tool call - returns chunks
                return [
                    {"messageStart": {"role": "assistant"}},
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
            # Synchronous full response with tool call
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

        # Simple text response for tools (when full_response=False)
        if stream:
            return ["I'll check the top song for WABC."]
        return "I'll check the top song for WABC."

    def _generate_text_response(self, stream: bool) -> str | dict | list:
        """
        Generate a mock response with regular text content.

        :param stream: Whether to stream the response
        :return: Text response in the appropriate format
        """
        if self.full_response:
            if stream:
                # Streamed full response - returns event chunks
                return [
                    {"messageStart": {"role": "assistant"}},
                    {"contentBlockDelta": {"delta": {"text": "Test response"}}},
                    {"messageStop": {"stopReason": "end_turn"}},
                ]
            # Synchronous full response with text
            return {
                "output": {"message": {"content": [{"text": "Test response"}]}},
                "usage": {"inputTokens": 10, "outputTokens": 20},
            }

        # Simple text response (when full_response=False)
        if stream:
            return ["Test response"]
        return "Test response"


class MockBedrockModelNoConfig(BedrockBase):
    """
    Mock model that initializes with minimal configuration.

    This class is used to test the behavior of BedrockBase when inferenceConfig
    and toolConfig are not explicitly provided.
    """

    def __init__(self, **kwargs):
        """
        Initialize a minimal mock model with only required parameters.

        :param kwargs: Additional keyword arguments to pass to the parent class
        """
        super().__init__(**kwargs)
        self.model_id = "MODEL_ID_HAIKU"


class TestBedrockBase:
    """
    Test suite for the BedrockBase class.

    This suite covers initialization, configuration, invocation patterns,
    and response handling for the BedrockBase class and its derivatives.
    """

    # -------------------------------------------------------------------------
    # Fixtures
    # -------------------------------------------------------------------------

    @pytest.fixture
    def common_params(self):
        """
        Return common parameters used across test fixtures.

        Having these parameters in a separate fixture reduces duplication
        and makes it easier to change them globally if needed.
        """
        return {
            "source": "test_source",
            "metric_prefix": "test_prefix",
            "extra_tags": {"test_tag": "value"},
        }

    @pytest.fixture
    def sample_tool_config(self):
        """
        Create a sample BedrockToolConfig object for testing.

        This fixture provides a reusable tool configuration that can be
        used across multiple tests.
        """
        return BedrockToolConfig(
            tools=[
                BedrockTool(
                    toolSpec=BedrockToolSpec(
                        name="top_song",
                        description="Get the most popular song played on a radio station.",
                        inputSchema=BedrockToolInputSchema(
                            json=BedrockJSONSchema(
                                type="object",
                                properties={
                                    "sign": {
                                        "type": "string",
                                        "description": "The call sign for the radio station",
                                    }
                                },
                                required=["sign"],
                            )
                        ),
                    )
                )
            ]
        )

    @pytest.fixture
    def bedrock_base(self, common_params):
        """
        Create a standard MockBedrockModel instance for testing.

        This fixture is the base case for most tests - a model with
        default configurations and full_response=False.
        """
        return MockBedrockModel(
            full_response=False,
            inferenceConfig=BedrockInferenceConfig(
                temperature=1,
                max_tokens=None,
            ),
            **common_params,
        )

    @pytest.fixture
    def bedrock_base_full_response(self, common_params):
        """
        Create a MockBedrockModel instance with full_response=True.

        This fixture is used to test the behavior when full API responses
        are returned instead of just completion text.
        """
        return MockBedrockModel(
            full_response=True,
            inferenceConfig=BedrockInferenceConfig(
                temperature=1,
                max_tokens=None,
            ),
            **common_params,
        )

    @pytest.fixture
    def bedrock_base_no_config(self, common_params):
        """
        Create a MockBedrockModelNoConfig instance with minimal configuration.

        This fixture is used to test the behavior when inferenceConfig and
        toolConfig are not explicitly provided.
        """
        return MockBedrockModelNoConfig(full_response=False, **common_params)

    @pytest.fixture
    def bedrock_base_with_tools(self, common_params, sample_tool_config):
        """
        Create a MockBedrockModel instance with tool configuration.

        This fixture is used to test the behavior when the model is
        configured with tools that can be invoked.
        """
        return MockBedrockModel(
            full_response=False,
            inferenceConfig=BedrockInferenceConfig(
                temperature=1,
                max_tokens=None,
            ),
            toolConfig=sample_tool_config,
            **common_params,
        )

    @pytest.fixture
    def bedrock_base_with_tools_full_response(self, common_params, sample_tool_config):
        """
        Create a MockBedrockModel with tools and full_response=True.

        This fixture is used to test the behavior when both tools are
        configured and full API responses are requested.
        """
        return MockBedrockModel(
            full_response=True,
            inferenceConfig=BedrockInferenceConfig(
                temperature=1,
                max_tokens=None,
            ),
            toolConfig=sample_tool_config,
            **common_params,
        )

    # -------------------------------------------------------------------------
    # Initialization Tests
    # -------------------------------------------------------------------------

    def test_initialization(self, bedrock_base):
        """
        Test that the BedrockBase initializes correctly with default settings.

        This test verifies:
        1. Basic attributes are set correctly
        2. inferenceConfig is initialized with default values
        3. toolConfig is correctly handled when a dictionary format is used
        """
        assert bedrock_base.source == "test_source"
        assert bedrock_base.metric_prefix == "test_prefix"
        assert bedrock_base.logging_tags == {"test_tag": "value"}
        assert bedrock_base.model_id == "MODEL_ID_HAIKU"

        # Validate inferenceConfig initialization
        assert isinstance(bedrock_base.inferenceConfig, BedrockInferenceConfig)
        assert bedrock_base.inferenceConfig.temperature == 1
        assert bedrock_base.inferenceConfig.max_tokens is None

        # Validate toolConfig handling
        assert bedrock_base.toolConfig is None

    def test_initialization_without_configs(self, bedrock_base_no_config):
        """
        Test BedrockBase initialization when no configs are provided.

        This test verifies that the model correctly initializes with None
        values for inferenceConfig and toolConfig when they are not provided.
        """
        assert bedrock_base_no_config.source == "test_source"
        assert bedrock_base_no_config.metric_prefix == "test_prefix"
        assert bedrock_base_no_config.logging_tags == {"test_tag": "value"}
        assert bedrock_base_no_config.model_id == "MODEL_ID_HAIKU"

        # Config values should be None when not provided
        assert bedrock_base_no_config.inferenceConfig is None
        assert bedrock_base_no_config.toolConfig is None

    def test_toolconfig_initialization_formats(self, common_params):
        """
        Test initialization with different toolConfig formats.

        This test verifies that the BedrockBase correctly handles three
        different formats for providing tool configurations:
        1. Dictionary format (mimicking JSON input)
        2. BedrockToolConfig object
        3. List of BedrockTool objects

        All three formats should result in a valid BedrockToolConfig object.
        """
        # Test with dictionary format (resembling parsed JSON)
        model_dict = MockBedrockModelNoConfig(
            toolConfig={
                "tools": [
                    {
                        "toolSpec": {
                            "name": "test_tool",
                            "description": "A test tool",
                            "inputSchema": {
                                "json": {
                                    "type": "object",
                                    "properties": {
                                        "param": {
                                            "type": "string",
                                            "description": "A test parameter",
                                        }
                                    },
                                    "required": ["param"],
                                }
                            },
                        }
                    }
                ]
            },
            **common_params,
        )
        assert isinstance(model_dict.toolConfig, BedrockToolConfig)
        assert len(model_dict.toolConfig.tools) == 1
        assert model_dict.toolConfig.tools[0].toolSpec.name == "test_tool"

        # Test with BedrockToolConfig object
        model_config = MockBedrockModelNoConfig(
            toolConfig=BedrockToolConfig(
                tools=[
                    BedrockTool(
                        toolSpec=BedrockToolSpec(
                            name="test_tool",
                            description="A test tool",
                            inputSchema=BedrockToolInputSchema(
                                json=BedrockJSONSchema(
                                    type="object",
                                    properties={
                                        "param": {
                                            "type": "string",
                                            "description": "A test parameter",
                                        }
                                    },
                                    required=["param"],
                                )
                            ),
                        )
                    )
                ]
            ),
            **common_params,
        )
        assert isinstance(model_config.toolConfig, BedrockToolConfig)
        assert len(model_config.toolConfig.tools) == 1
        assert model_config.toolConfig.tools[0].toolSpec.name == "test_tool"

        # Test with list of tools format
        model_list = MockBedrockModelNoConfig(
            toolConfig=[
                BedrockTool(
                    toolSpec=BedrockToolSpec(
                        name="test_tool",
                        description="A test tool",
                        inputSchema=BedrockToolInputSchema(
                            json=BedrockJSONSchema(
                                type="object",
                                properties={
                                    "param": {
                                        "type": "string",
                                        "description": "A test parameter",
                                    }
                                },
                                required=["param"],
                            )
                        ),
                    )
                )
            ],
            **common_params,
        )
        assert isinstance(model_list.toolConfig, BedrockToolConfig)
        assert len(model_list.toolConfig.tools) == 1
        assert model_list.toolConfig.tools[0].toolSpec.name == "test_tool"

    # -------------------------------------------------------------------------
    # Basic Invocation Tests
    # -------------------------------------------------------------------------

    def test_invoke_with_different_prompt_types(self, bedrock_base):
        """
        Test invocation with different prompt input types.

        This test verifies that the model can handle both string prompts
        and structured Messages objects as input.
        """
        # Test with string prompt
        str_response = bedrock_base.invoke("Test prompt")
        assert str_response == "Test response"

        # Test with Messages prompt
        messages = Messages(messages=[{"role": "user", "content": "Test prompt"}])
        msg_response = bedrock_base.invoke(messages)
        assert msg_response == "Test response"

    def test_invoke_with_override_kwargs(self, bedrock_base):
        """
        Test invoking with override kwargs.

        This test verifies that the model accepts additional keyword arguments
        when invoking. In a real API call, these would override the default
        model configuration parameters.
        """
        response = bedrock_base.invoke("Test prompt", temperature=0.7, maxTokens=1024)
        assert response == "Test response"

    def test_invalid_prompt_type(self, bedrock_base):
        """
        Test error handling with invalid prompt types.

        This test verifies that the model correctly raises a ValueError
        when an unsupported prompt type is provided.
        """
        with pytest.raises(
            ValueError, match="Prompt must be a string or a Messages object!"
        ):
            bedrock_base.invoke(123)  # Invalid type

    # -------------------------------------------------------------------------
    # Full Response Tests
    # -------------------------------------------------------------------------

    def test_invoke_with_full_response(self, bedrock_base_full_response):
        """
        Test invocation with full_response=True.

        This test verifies that when full_response=True, the model returns
        the complete API response object rather than just the completion text.
        """
        response = bedrock_base_full_response.invoke("Test prompt")
        assert isinstance(response, dict)
        assert response["output"]["message"]["content"][0]["text"] == "Test response"
        assert response["usage"]["inputTokens"] == 10
        assert response["usage"]["outputTokens"] == 20

    # -------------------------------------------------------------------------
    # Tool Tests
    # -------------------------------------------------------------------------

    def test_invoke_with_tools(self, bedrock_base_with_tools):
        """
        Test invocation with tools configured.

        This test verifies that when tools are configured, the model returns
        an appropriate response indicating tool use.
        """
        response = bedrock_base_with_tools.invoke("Test prompt")
        assert response == "I'll check the top song for WABC."

    def test_invoke_with_tools_full_response(
        self, bedrock_base_with_tools_full_response
    ):
        """
        Test invocation with tools and full_response=True.

        This test verifies that when both tools are configured and full_response=True,
        the model returns the complete API response object with tool call information.
        """
        response = bedrock_base_with_tools_full_response.invoke("Test prompt")
        assert isinstance(response, dict)
        assert "toolCall" in response["output"]["message"]["content"][0]
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

    # -------------------------------------------------------------------------
    # Streaming Tests
    # -------------------------------------------------------------------------

    def test_stream_with_tools(self, bedrock_base_with_tools):
        """
        Test streaming with tools configured.

        This test verifies that when tools are configured and streaming is used,
        the model returns a stream of response chunks with tool information.
        """
        chunks = list(bedrock_base_with_tools.stream("Test prompt"))
        assert len(chunks) == 1
        assert chunks[0] == "I'll check the top song for WABC."

    def test_stream_with_tools_full_response(
        self, bedrock_base_with_tools_full_response
    ):
        """
        Test streaming with tools and full_response=True.

        This test verifies that when both tools are configured, streaming is used,
        and full_response=True, the model returns a stream of API event objects.
        """
        chunks = list(bedrock_base_with_tools_full_response.stream("Test prompt"))
        assert len(chunks) == 3  # messageStart, contentBlockDelta, messageStop
        assert chunks[0]["messageStart"]["role"] == "assistant"
        assert "toolCall" in chunks[1]["contentBlockDelta"]["delta"]
        assert chunks[1]["contentBlockDelta"]["delta"]["toolCall"]["name"] == "top_song"
        assert chunks[2]["messageStop"]["stopReason"] == "tool_call"

    def test_stream_without_tools(self, bedrock_base):
        """
        Test streaming without tools.

        This test verifies that when streaming is used without tools,
        the model returns a stream of text chunks.
        """
        chunks = list(bedrock_base.stream("Test prompt"))
        assert len(chunks) == 1
        assert chunks[0] == "Test response"

    def test_stream_without_tools_full_response(self, bedrock_base_full_response):
        """
        Test streaming without tools and full_response=True.

        This test verifies that when streaming is used without tools and
        full_response=True, the model returns a stream of API event objects.
        """
        chunks = list(bedrock_base_full_response.stream("Test prompt"))
        assert len(chunks) == 3  # messageStart, contentBlockDelta, messageStop
        assert chunks[0]["messageStart"]["role"] == "assistant"
        assert chunks[1]["contentBlockDelta"]["delta"]["text"] == "Test response"
        assert chunks[2]["messageStop"]["stopReason"] == "end_turn"

    # -------------------------------------------------------------------------
    # Real Bedrock Model Tests
    # -------------------------------------------------------------------------

    @pytest.mark.skipif(
        not has_aws_credentials,
        reason="AWS credentials not found in configured profiles",
    )
    def test_real_bedrock_invoke(self):
        """
        Test invoking a real Bedrock model with a simple prompt.
        """
        from fence.models.bedrock.claude import Claude35Sonnet

        model = Claude35Sonnet(source="test")
        response = model.invoke("What is 2+2?")
        assert isinstance(response, str)
        assert "4" in response

    @pytest.mark.skipif(
        not has_aws_credentials,
        reason="AWS credentials not found in configured profiles",
    )
    def test_real_bedrock_stream(self):
        """
        Test streaming from a real Bedrock model.
        """
        from fence.models.bedrock.claude import Claude35Sonnet

        model = Claude35Sonnet(source="test")
        chunks = list(model.stream("Count from 1 to 3."))
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)
        assert any("1" in chunk for chunk in chunks)
        assert any("2" in chunk for chunk in chunks)
        assert any("3" in chunk for chunk in chunks)

    @pytest.mark.skipif(
        not has_aws_credentials,
        reason="AWS credentials not found in configured profiles",
    )
    def test_real_bedrock_full_response(self):
        """
        Test getting a full response from a real Bedrock model.
        """
        from fence.models.bedrock.claude import Claude35Sonnet

        model = Claude35Sonnet(source="test", full_response=True)
        response = model.invoke("What is the capital of France?")
        assert isinstance(response, dict)
        assert "output" in response
        assert "message" in response["output"]
        assert "content" in response["output"]["message"]
        assert any(
            "Paris" in content.get("text", "")
            for content in response["output"]["message"]["content"]
        )
        assert "usage" in response
        assert "inputTokens" in response["usage"]
        assert "outputTokens" in response["usage"]

    @pytest.mark.skipif(
        not has_aws_credentials,
        reason="AWS credentials not found in configured profiles",
    )
    def test_real_bedrock_with_tools(self):
        """
        Test using a real Bedrock model with tools.
        """
        from fence.models.bedrock.base import (
            BedrockJSONSchema,
            BedrockTool,
            BedrockToolConfig,
            BedrockToolInputSchema,
            BedrockToolSpec,
        )
        from fence.models.bedrock.claude import Claude35Sonnet

        # Create a simple calculator tool
        calculator_tool = BedrockTool(
            toolSpec=BedrockToolSpec(
                name="calculator",
                description="A simple calculator that can add two numbers",
                inputSchema=BedrockToolInputSchema(
                    json=BedrockJSONSchema(
                        type="object",
                        properties={
                            "a": {"type": "number", "description": "First number"},
                            "b": {"type": "number", "description": "Second number"},
                        },
                        required=["a", "b"],
                    )
                ),
            )
        )

        model = Claude35Sonnet(
            source="test",
            full_response=True,
            toolConfig=BedrockToolConfig(tools=[calculator_tool]),
        )

        response = model.invoke("Add 5 and 3 using the calculator tool.")
        assert isinstance(response, dict)
        assert "output" in response
        assert "message" in response["output"]
        assert "content" in response["output"]["message"]

        # Print the response for debugging
        print(f"Response: {response}")

        # Check if any content block contains a toolUse or toolCall
        has_tool_use = any(
            "toolUse" in content or "toolCall" in content
            for content in response["output"]["message"]["content"]
        )
        assert has_tool_use, "Expected tool use in response"

        # Check if the tool name is mentioned in any text block
        has_calculator_mention = any(
            "calculator" in content.get("text", "").lower()
            for content in response["output"]["message"]["content"]
        )
        assert has_calculator_mention, "Expected calculator tool to be mentioned"

        # Check if the numbers 5 and 3 are mentioned in any text block
        has_numbers = any(
            "5" in content.get("text", "") and "3" in content.get("text", "")
            for content in response["output"]["message"]["content"]
        )
        assert has_numbers, "Expected numbers 5 and 3 to be mentioned"
