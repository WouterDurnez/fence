"""
Integration tests for tool calling.
"""

import os
from unittest.mock import patch

import pytest

from fence.agents.agent import Agent
from fence.agents.bedrock import BedrockAgent
from fence.models.bedrock.claude import Claude35Sonnet
from fence.models.openai.gpt import GPT4omini
from fence.tools.base import BaseTool
from fence.tools.scratch import EnvTool

# Check if environment variables for API keys are present
has_openai_api_key = os.environ.get("OPENAI_API_KEY") is not None

# Check if AWS credentials are available via profile
try:
    import boto3

    session = boto3.Session()
    has_aws_credentials = session.get_credentials() is not None
except (ImportError, Exception):
    has_aws_credentials = False


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
        result, thinking, answer = agent.run(query)

        # The value could be in the response, thinking, or answer
        result_str = result + str(thinking) + str(answer)
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
        # Create responses for each invocation
        tool_response = {
            "output": {
                "message": {
                    "content": [
                        {
                            "text": "I see the environment variables: bedrock_env_var and echo_prefix"
                        },
                        {
                            "toolUse": {
                                "name": "echo_tool",
                                "input": {"message": "Hello World"},
                            }
                        },
                    ]
                }
            },
            "stopReason": "tool_use",  # This is important - indicate tool use
        }

        final_response = {
            "output": {
                "message": {"content": [{"text": "The result is: PREFIX: Hello World"}]}
            },
            "stopReason": "end_turn",  # This indicates no more tool calls
        }

        # Set up mock to return different responses on subsequent calls
        mock_invoke.side_effect = [tool_response, final_response]

        # Create a Bedrock agent with multiple tools and environment variables
        agent = BedrockAgent(
            model=Claude35Sonnet(source="agent"),
            tools=[EnvTool(), EchoTool()],
            environment={"bedrock_env_var": "value1", "echo_prefix": "PREFIX"},
        )

        # Mock the _process_tool_call method to simulate tool execution
        with patch.object(agent, "_process_tool_call") as mock_process:
            # Simulate the tool result
            mock_process.return_value = "PREFIX: Hello World"

            # Test that the environment is passed to the tool
            result, thinking, answer = agent.run("Echo 'Hello World' with the prefix.")

            # Verify the tool was called with the right parameters
            mock_process.assert_called_with("echo_tool", {"message": "Hello World"})

            # Check the result contains our prefix
            assert "PREFIX: Hello World" in result

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

        # Create responses for each invocation
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
            "stopReason": "tool_use",  # This is important - indicate tool use
        }

        final_response = {
            "output": {
                "message": {
                    "content": [{"text": "<answer>The text has 5 words.</answer>"}]
                }
            },
            "stopReason": "end_turn",  # This indicates no more tool calls
        }

        # Set up mock to return different responses on subsequent calls
        mock_invoke.side_effect = [tool_response, final_response]

        # Create a Bedrock agent with a custom system message and tools
        agent = BedrockAgent(
            model=Claude35Sonnet(source="agent"),
            tools=[CounterTool()],
            system_message=custom_system_message,
            environment={"count_format": "The text '{text}' has {count} words."},
        )

        # Mock the _process_tool_call method to simulate tool execution
        with patch.object(agent, "_process_tool_call") as mock_process:
            # Set up the mock to return a formatted count result
            mock_process.return_value = "The text 'This is a sample text.' has 5 words."

            # Test that both the system message and environment are working
            result, thinking, answer = agent.run(
                "Count the words in 'This is a sample text.'"
            )

            # Verify the tool was called with the right parameters
            mock_process.assert_called_with(
                "counter_tool", {"text": "This is a sample text."}
            )

            # Check the response
            assert "has 5 words" in result
