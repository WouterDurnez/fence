"""
Integration tests for tool calling.
"""

from fence.agents.agent import Agent
from fence.agents.bedrock import BedrockAgent
from fence.models.bedrock.claude import Claude35Sonnet
from fence.models.openai.gpt import GPT4omini
from fence.tools.base import BaseTool
from fence.tools.scratch import EnvTool


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

        assert "bedrock_env_var" in result
        assert "bedrock_value" in result

    def test_bedrock_multiple_tools_with_environment(self):
        """
        Test multiple tools with environment variables using a Bedrock agent.
        """
        # Create a Bedrock agent with multiple tools and environment variables
        agent = BedrockAgent(
            model=Claude35Sonnet(source="agent"),
            tools=[EnvTool(), EchoTool()],
            environment={
                "bedrock_env_var": "bedrock_value",
                "echo_prefix": "ECHO_RESPONSE",
            },
        )

        query = """First, tell me what environment variables are available.
                   Then, use the echo tool to echo back the message 'Hello World'."""
        result = agent.run(query)

        # Check that both tools were used correctly with environment variables
        assert "bedrock_env_var" in result
        assert "bedrock_value" in result
        assert "echo_prefix" in result
        assert "ECHO_RESPONSE" in result
        assert "Hello World" in result

    def test_bedrock_system_message_with_tools(self):
        """
        Test that the Bedrock agent follows system message instructions when using tools.
        """
        # Create a Bedrock agent with system message, tools, and environment variables
        system_message = """You are a helpful assistant that analyzes text.
        Always format your final response with a summary section and a details section.
        Always count words before responding."""

        agent = BedrockAgent(
            model=Claude35Sonnet(source="agent"),
            tools=[CounterTool()],
            system_message=system_message,
            environment={
                "count_format": "ANALYSIS: The text '{text}' has {count} words."
            },
        )

        query = "Analyze this text: The quick brown fox jumps over the lazy dog."
        result = agent.run(query)

        # Check that the agent followed the system message and used the tool with environment
        assert "ANALYSIS" in result
        assert "The quick brown fox jumps over the lazy dog" in result
        assert "9 words" in result
        assert "summary" in result.lower()
        assert "details" in result.lower()
