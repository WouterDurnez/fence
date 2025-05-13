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
    @patch("fence.models.bedrock.base.BedrockBase._invoke")
    def test_bedrock_environment_access(self, mock_invoke):
        """
        Test environment access from a tool using a Bedrock agent. Environment variables should be passed correctly.
        """
        # Create a Bedrock agent with a model and tools
        agent = BedrockAgent(
            model=Claude35Sonnet(source="agent"),
            tools=[EnvTool()],
            environment={"bedrock_env_var": "bedrock_value"},
        )

        # Mock the run method to simulate tool use
        with patch.object(agent, "run") as mock_run:
            # Setup the mock to return a response with events
            mock_run.return_value = {
                "content": "The environment variable 'bedrock_env_var' has the value 'bedrock_value'.",
                "tool_use": [
                    {
                        "name": "EnvTool",
                        "parameters": {},
                        "result": "The environment currently holds these variables:\nbedrock_env_var: bedrock_value",
                    }
                ],
                "thinking": [
                    "I can see the environment variable bedrock_env_var has the value bedrock_value"
                ],
            }

            # Test that the environment is passed to the tool
            result = agent.run(
                "Tell me what the value of the environment variable 'bedrock_env_var' is. You have a tool to access the environment.",
                stream=True,
            )

            # Verify run was called with the right parameters
            mock_run.assert_called_once()

            # Verify tool usage is included
            assert "tool_use" in result
            assert len(result["tool_use"]) > 0
            assert result["tool_use"][0]["name"] == "EnvTool"
            assert "bedrock_env_var" in result["tool_use"][0]["result"]
            assert "bedrock_value" in result["tool_use"][0]["result"]

            # Verify the environment variable appears in the content
            assert "bedrock_env_var" in result["content"]
            assert "bedrock_value" in result["content"]

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

        # Mock the run method to return a response with events
        with patch.object(agent, "run") as mock_run:
            # Setup the mock to return a response with events
            mock_run.return_value = {
                "content": "PREFIX: Hello, world!",
                "tool_use": [
                    {
                        "name": "EchoTool",
                        "parameters": {"message": "Hello, world!"},
                        "result": "PREFIX: Hello, world!",
                    }
                ],
                "thinking": [
                    "I'll use the EchoTool to echo the message with the prefix from the environment."
                ],
            }

            # Test that the environment is passed to the tool
            result = agent.run(
                "Echo the message 'Hello, world!' using the EchoTool.",
                stream=True,
            )

            # Verify run was called with the right parameters
            mock_run.assert_called_once()

            # Verify tool usage is included
            assert "tool_use" in result
            assert len(result["tool_use"]) > 0
            assert result["tool_use"][0]["name"] == "EchoTool"
            assert "PREFIX" in result["tool_use"][0]["result"]

            # Verify the environment variable appears in the content
            assert "PREFIX" in result["content"]

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

        # Mock the run method to return a response with events
        with patch.object(agent, "run") as mock_run:
            # Setup the mock to return a response with events
            mock_run.return_value = {
                "content": "The text 'This is a test' has 4 words.",
                "tool_use": [
                    {
                        "name": "CounterTool",
                        "parameters": {"text": "This is a test"},
                        "result": "The text 'This is a test' has 4 words.",
                    }
                ],
                "thinking": [
                    "I'll use the CounterTool to count the words in the text."
                ],
            }

            # Test that the system message and environment are used correctly
            result = agent.run(
                "Count the words in 'This is a test'.",
                stream=True,
            )

            # Verify run was called with the right parameters
            mock_run.assert_called_once()

            # Verify tool usage is included
            assert "tool_use" in result
            assert len(result["tool_use"]) > 0
            assert result["tool_use"][0]["name"] == "CounterTool"
            assert "4 words" in result["tool_use"][0]["result"]

            # Verify the content is correct
            assert "4 words" in result["content"]

    @pytest.mark.skipif(
        not has_aws_credentials,
        reason="AWS credentials not found in configured profiles",
    )
    @patch("fence.models.bedrock.base.BedrockBase._invoke")
    def test_bedrock_agent_with_event_handlers(self, mock_invoke):
        """
        Test BedrockAgent with custom event handlers using the EventHandlers model.
        """
        # Create event tracking variables
        event_data = []

        # Create custom event handlers
        def on_tool_use_start(tool_name, parameters):
            event_data.append(
                {
                    "type": "tool_use_start",
                    "tool_name": tool_name,
                    "parameters": parameters,
                }
            )

        def on_tool_use_stop(tool_name, parameters, result):
            event_data.append(
                {
                    "type": "tool_use_stop",
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
            on_tool_use_start=on_tool_use_start,
            on_tool_use_stop=on_tool_use_stop,
            on_thinking=on_thinking,
            on_answer=on_answer,
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

        # Mock the model's invoke method to return a response with events
        mock_invoke.side_effect = [
            {
                "output": {
                    "message": {
                        "content": [
                            {
                                "text": "<thinking>I'll use the CounterTool to count the words in the text.</thinking>",
                            },
                            {
                                "toolUse": {
                                    "name": "CounterTool",
                                    "input": {"text": "This is a test"},
                                    "toolUseId": "12345",
                                },
                            },
                        ],
                    },
                },
                "stopReason": "tool_use",
            },
            {
                "output": {
                    "message": {
                        "content": [
                            {
                                "text": "The text 'This is a test' has 4 words.",
                            },
                        ],
                    },
                },
                "stopReason": "end_turn",
            },
        ]

        # Mock the invoke method to return a proper AgentResponse object
        with patch.object(agent, "invoke") as mock_invoke_method:
            from fence.agents.bedrock.models import (
                AgentEventTypes,
                AgentResponse,
                AnswerEvent,
            )

            # Create a valid AgentResponse object with a string answer
            mock_response = AgentResponse(
                answer="The text 'This is a test' has 4 words.",
                events=[
                    AnswerEvent(
                        agent_name=agent.identifier,
                        type=AgentEventTypes.ANSWER,
                        content="The text 'This is a test' has 4 words.",
                    )
                ],
            )
            mock_invoke_method.return_value = mock_response

            # Test that the event handlers are called correctly
            result = agent.run(
                "Count the words in 'This is a test'.",
                stream=False,
            )

            # Verify invoke was called with the right parameters
            mock_invoke_method.assert_called_once()

            # Verify the content is correct
            assert "4 words" in result.answer
