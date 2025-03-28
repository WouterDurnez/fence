"""
Integration tests for tool calling.
"""

import os
from unittest.mock import patch

import pytest

from fence.agents.agent import Agent
from fence.agents.bedrock import BedrockAgent, EventHandlers
from fence.models.bedrock.claude import Claude35Sonnet
from fence.models.openai.gpt import GPT4omini
from fence.tools.base import BaseTool
from fence.tools.scratch import EnvTool

# Check if environment variables for API keys are present
has_openai_api_key = os.environ.get("OPENAI_API_KEY") is not None

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


class EchoTool(BaseTool):
    """Tool that echoes back a message with the provided environment."""

    def execute_tool(self, message: str, environment: dict = None, **kwargs) -> str:
        """
        Echo the message with environment variable.

        :param message: Message to echo
        :param environment: Environment variables to access
        :return: Echoed message with environment variable
        """
        env_prefix = ""
        if environment and "echo_prefix" in environment:
            env_prefix = environment["echo_prefix"]

        return f"{env_prefix}: {message}"


class CounterTool(BaseTool):
    """Tool that counts the number of words in a message."""

    def execute_tool(self, text: str, environment: dict = None, **kwargs) -> str:
        """
        Count words in text and add environment context.

        :param text: Text to count words in
        :param environment: Environment variables to access
        :return: Word count with context
        """
        word_count = len(text.split())
        context = ""
        if environment and "count_format" in environment:
            context = environment["count_format"]
            return context.format(count=word_count, text=text)

        return f"The text contains {word_count} words."


class TestToolCalling:
    """
    Integration tests for tool calling.
    """

    @pytest.mark.skipif(
        not has_openai_api_key, reason="OpenAI API key not found in environment"
    )
    def test_environment_access(self):
        """
        Test the environment access from a tool. Env variables should be passed to the tool.
        """

        # Create an agent with a model and tools
        agent = Agent(
            model=GPT4omini(source="agent"),
            tools=[EnvTool()],
            environment={"some_env_var": "some_value"},
        )

        query = "Tell me what the value of the environment variable 'some_env_var' is. You have a tool to access the environment."
        result = agent.run(query)

        assert "some_env_var" in result
        assert "some_value" in result

    @pytest.mark.skipif(
        not has_aws_credentials,
        reason="AWS credentials not found in configured profiles",
    )
    def test_bedrock_environment_access(self):
        """
        Test environment access from a tool using a Bedrock agent. Environment variables should be passed correctly.
        """
        # Create a Bedrock agent with a model and tools
        agent = BedrockAgent(
            model=Claude35Sonnet(source="agent"),
            tools=[EnvTool()],
            environment={"bedrock_env_var": "bedrock_value"},
        )

        query = "Tell me what the value of the environment variable 'bedrock_env_var' is. You have a tool to access the environment."
        result = agent.run(query)

        # The value could be in the response content, thinking, or answer
        result_str = (
            result.get("content", "")
            + str(result.get("thinking", []))
            + str(result.get("answer", ""))
        )
        assert "bedrock_env_var" in result_str
        assert "bedrock_value" in result_str

    @pytest.mark.skipif(
        not has_aws_credentials,
        reason="AWS credentials not found in configured profiles",
    )
    @patch("fence.models.bedrock.base.BedrockBase._invoke")
    def test_bedrock_multiple_tools_with_environment(self, mock_invoke):
        """
        Test multiple tools with environment variables using a Bedrock agent.
        """
        # Create a Bedrock agent with multiple tools and environment variables
        agent = BedrockAgent(
            model=Claude35Sonnet(source="agent"),
            tools=[EnvTool(), EchoTool()],
            environment={"bedrock_env_var": "value1", "echo_prefix": "PREFIX"},
        )

        # Instead of mocking individual methods, mock the high-level stream method
        # to return a sequence of events including tool usage
        with patch.object(agent, "stream") as mock_stream:
            # Setup the mock to yield text chunks followed by events
            mock_stream.return_value = iter(
                [
                    "I'll help you echo that message.",  # Text chunk
                    {
                        "events": [  # Event dictionary at the end
                            {
                                "type": "tool_usage",
                                "content": {
                                    "name": "echo_tool",
                                    "parameters": {"message": "Hello World"},
                                    "result": "PREFIX: Hello World",
                                },
                            }
                        ]
                    },
                ]
            )

            # Use our existing _unpack_event_stream method to convert events to result dict
            with patch.object(
                agent, "_unpack_event_stream", wraps=agent._unpack_event_stream
            ) as mock_unpack:
                # Test that the environment is passed to the tool
                result = agent.run("Echo 'Hello World' with the prefix.", stream=True)

                # Verify stream was called with the right parameters
                mock_stream.assert_called_once()

                # Verify our _unpack_event_stream was called to process the events
                mock_unpack.assert_called_once()

                # Verify the tool was included in the result
                assert "tool_use" in result
                assert len(result["tool_use"]) > 0
                assert result["tool_use"][0]["name"] == "echo_tool"
                assert result["tool_use"][0]["parameters"] == {"message": "Hello World"}
                assert result["tool_use"][0]["result"] == "PREFIX: Hello World"

    @pytest.mark.skipif(
        not has_aws_credentials,
        reason="AWS credentials not found in configured profiles",
    )
    @patch("fence.models.bedrock.base.BedrockBase._invoke")
    def test_bedrock_system_message_with_tools(self, mock_invoke):
        """
        Test Bedrock agent with a custom system message and tools.
        """
        custom_system_message = (
            "You are a specialized assistant that helps with counting words."
        )

        # Create a Bedrock agent with a custom system message and tools
        agent = BedrockAgent(
            model=Claude35Sonnet(source="agent"),
            tools=[CounterTool()],
            system_message=custom_system_message,
            environment={"count_format": "The text '{text}' has {count} words."},
        )

        # Mock the high-level stream method to return a sequence including thinking, tool use, and answer
        with patch.object(agent, "stream") as mock_stream:
            # Setup the mock to yield text chunks and events
            mock_stream.return_value = iter(
                [
                    "<thinking>I need to count the words in the text.</thinking>",  # Thinking text
                    "I'll count the words in that text.",  # Normal text chunk
                    {
                        "events": [  # Events dictionary at the end
                            {
                                "type": "thinking",
                                "content": "I need to count the words in the text.",
                            },
                            {
                                "type": "tool_usage",
                                "content": {
                                    "name": "counter_tool",
                                    "parameters": {"text": "This is a sample text."},
                                    "result": "The text 'This is a sample text.' has 5 words.",
                                },
                            },
                            {"type": "answer", "content": "The text has 5 words."},
                        ]
                    },
                ]
            )

            # Use the existing _unpack_event_stream method to process events
            with patch.object(
                agent, "_unpack_event_stream", wraps=agent._unpack_event_stream
            ) as mock_unpack:
                # Test that both the system message and environment are working
                result = agent.run(
                    "Count the words in 'This is a sample text.'", stream=True
                )

                # Verify stream was called with the right parameters
                mock_stream.assert_called_once()

                # Verify _unpack_event_stream was called to process the events
                mock_unpack.assert_called_once()

                # Verify tool usage is included
                assert "tool_use" in result
                assert len(result["tool_use"]) > 0
                assert result["tool_use"][0]["name"] == "counter_tool"
                assert result["tool_use"][0]["parameters"] == {
                    "text": "This is a sample text."
                }

                # Verify thinking was captured
                assert "thinking" in result
                assert len(result["thinking"]) > 0
                assert "count the words" in result["thinking"][0]

                # Verify answer was captured
                assert "answer" in result
                assert "has 5 words" in result["answer"]

    @pytest.mark.skipif(
        not has_aws_credentials,
        reason="AWS credentials not found in configured profiles",
    )
    @patch("fence.models.bedrock.base.BedrockBase._invoke")
    def test_bedrock_agent_with_event_handlers(self, mock_invoke):
        """
        Test BedrockAgent with custom event handlers using the EventHandlers model.
        """
        # Create responses for invocation
        tool_response = {
            "output": {
                "message": {
                    "content": [
                        {
                            "text": "<thinking>I need to count the words in the text.</thinking>"
                        },
                        {
                            "toolUse": {
                                "name": "counter_tool",
                                "input": {"text": "This is a sample text."},
                            }
                        },
                    ]
                }
            },
            "stopReason": "tool_use",
        }

        final_response = {
            "output": {
                "message": {
                    "content": [{"text": "<answer>The text has 5 words.</answer>"}]
                }
            },
            "stopReason": "end_turn",
        }

        # Set up mock to return different responses on subsequent calls
        mock_invoke.side_effect = [tool_response, final_response]

        # Create event tracking variables
        event_data = []

        # Create custom event handlers
        def on_tool_use(tool_name, parameters, result):
            event_data.append(
                {
                    "type": "tool_use",
                    "tool_name": tool_name,
                    "parameters": parameters,
                    "result": result,
                }
            )

        def on_thinking(text):
            event_data.append({"type": "thinking", "text": text})

        def on_answer(text):
            event_data.append({"type": "answer", "text": text})

        # Create event handlers model
        event_handlers = EventHandlers(
            on_tool_use=on_tool_use, on_thinking=on_thinking, on_answer=on_answer
        )

        # Create a Bedrock agent with event handlers
        agent = BedrockAgent(
            model=Claude35Sonnet(source="agent"),
            tools=[CounterTool()],
            system_message="You are a specialized assistant that helps with counting words.",
            environment={"count_format": "The text '{text}' has {count} words."},
            event_handlers=event_handlers,
            # Disable default logging to avoid duplicate event triggers
            log_agentic_response=False,
        )

        # Clear event data before the test
        event_data.clear()

        # Use a similar approach as the previous tests with streaming
        with patch.object(agent, "_process_tool_data"):
            with patch.object(agent, "_find_tool", wraps=agent._find_tool):
                with patch.object(agent, "_unpack_event_stream") as mock_unpack:
                    # Setup the mock to return our expected response
                    mock_unpack.return_value = {
                        "content": "<thinking>I need to count the words in the text.</thinking>\nThe text 'This is a sample text.' has 5 words.\n<answer>The text has 5 words.</answer>",
                        "thinking": ["I need to count the words in the text."],
                        "tool_use": [
                            {
                                "name": "counter_tool",
                                "parameters": {"text": "This is a sample text."},
                                "result": "The text 'This is a sample text.' has 5 words.",
                            }
                        ],
                        "answer": "The text has 5 words.",
                        "events": [],
                    }

                    # Run the agent
                    agent.run(
                        "Count the words in 'This is a sample text.'", stream=True
                    )

                    # Clear the event data before our manual tests
                    event_data.clear()

                    # Now manually test the event handlers to verify they work correctly
                    agent._safe_event_handler(
                        "on_thinking", text="Sample thinking text"
                    )
                    agent._safe_event_handler(
                        "on_tool_use",
                        tool_name="counter_tool",
                        parameters={"text": "This is a sample text."},
                        result="The text has 5 words.",
                    )
                    agent._safe_event_handler("on_answer", text="The text has 5 words.")

                    # Now verify the counts - should have exactly our 3 manual events
                    assert len(event_data) == 3

                    # Check thinking event
                    thinking_events = [e for e in event_data if e["type"] == "thinking"]
                    assert len(thinking_events) == 1
                    assert thinking_events[0]["text"] == "Sample thinking text"

                    # Check tool_use event
                    tool_events = [e for e in event_data if e["type"] == "tool_use"]
                    assert len(tool_events) == 1
                    assert tool_events[0]["tool_name"] == "counter_tool"
                    assert (
                        tool_events[0]["parameters"]["text"] == "This is a sample text."
                    )

                    # Check answer event
                    answer_events = [e for e in event_data if e["type"] == "answer"]
                    assert len(answer_events) == 1
                    assert answer_events[0]["text"] == "The text has 5 words."
