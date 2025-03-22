"""
Tests for the BedrockAgent class.

This module contains tests for the BedrockAgent implementation, including:
1. Initialization with different configurations
2. Memory management and flushing
3. Tool registration and calling
4. Response handling with various formats
5. Stream and invoke methodologies
"""

from unittest.mock import Mock, patch

import pytest

from fence.agents.base import AgentLogType
from fence.agents.bedrock import BedrockAgent
from fence.memory.base import FleetingMemory
from fence.models.base import LLM
from fence.models.bedrock.base import BedrockTool
from fence.tools.base import BaseTool


class MockLLM(LLM):
    """Mock LLM implementation for testing BedrockAgent."""

    def __init__(self, response_type="text", **kwargs):
        """
        Initialize the mock LLM.

        :param response_type: Type of response to return ("text", "tool_call", or "error")
        :param kwargs: Additional arguments to pass to the parent class
        """
        super().__init__(**kwargs)
        self.full_response = False
        self.model_name = "mock_llm"
        self.response_type = response_type
        self.invoke_called = False
        self.stream_called = False
        self.toolConfig = None

    def invoke(self, prompt, **kwargs):
        """Mock implementation of invoke."""
        self.invoke_called = True

        if self.response_type == "error":
            raise Exception("Test error")

        if self.response_type == "tool_call":
            if self.full_response:
                return {
                    "output": {
                        "message": {
                            "content": [
                                {"text": "I'll help with that."},
                                {
                                    "toolUse": {
                                        "name": "get_weather",
                                        "input": {"location": "New York"},
                                    }
                                },
                            ]
                        }
                    },
                    "stopReason": "tool_use",
                }
            return "I'll help with that using the get_weather tool."

        # Default text response
        if self.full_response:
            return {
                "output": {
                    "message": {"content": [{"text": "This is a test response."}]}
                },
                "stopReason": "end_turn",
            }
        return "This is a test response."

    def stream(self, prompt, **kwargs):
        """Mock implementation of stream."""
        self.stream_called = True

        if self.response_type == "error":
            raise Exception("Test error")

        if self.response_type == "tool_call":
            if self.full_response:
                yield {"messageStart": {"role": "assistant"}}
                yield {"contentBlockDelta": {"delta": {"text": "I'll help with that."}}}
                yield {
                    "contentBlockStart": {
                        "start": {
                            "toolUse": {
                                "name": "get_weather",
                                "arguments": {"location": "New York"},
                            }
                        }
                    }
                }
                yield {"contentBlockStop": {}}
                yield {"messageStop": {"stopReason": "tool_use"}}
            else:
                yield "I'll help with that using the get_weather tool."
        else:
            # Default text response
            if self.full_response:
                yield {"messageStart": {"role": "assistant"}}
                yield {"contentBlockDelta": {"delta": {"text": "This is a "}}}
                yield {"contentBlockDelta": {"delta": {"text": "test response."}}}
                yield {"messageStop": {"stopReason": "end_turn"}}
            else:
                yield "This is a "
                yield "test response."


class MockTool(BaseTool):
    """Mock tool implementation for testing BedrockAgent."""

    def __init__(self, name="test_tool", description="A test tool"):
        """Initialize the mock tool."""
        self._name = name
        self._description = description
        self.run_called = False
        self.run_args = None

    def execute_tool(self, environment: dict = None, **kwargs):
        """
        Execute the tool with the given parameters.

        :param environment: Dictionary containing environment variables
        :param kwargs: Additional parameters for the tool
        :return: A formatted result string
        """
        return f"Result from {self._name}"

    def run(self, **kwargs):
        """Mock implementation of run."""
        self.run_called = True
        self.run_args = kwargs
        return f"Result from {self._name}"

    def get_tool_name(self):
        """Return the tool name."""
        return self._name

    def get_tool_description(self):
        """Return the tool description."""
        return self._description

    def model_dump_bedrock_converse(self):
        """Return a mock model dump for Bedrock Converse."""
        return {
            "toolSpec": {
                "name": self._name,
                "description": self._description,
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The location",
                            }
                        },
                        "required": ["location"],
                    }
                },
            }
        }


class TestBedrockAgent:
    """Test suite for the BedrockAgent class."""

    # -------------------------------------------------------------------------
    # Fixtures
    # -------------------------------------------------------------------------

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM for testing."""
        return MockLLM()

    @pytest.fixture
    def mock_tool(self):
        """Create a mock tool for testing."""
        return MockTool()

    @pytest.fixture
    def memory(self):
        """Create a memory instance for testing."""
        return FleetingMemory()

    @pytest.fixture
    def agent(self, mock_llm, memory):
        """Create a BedrockAgent instance with a mock LLM and memory."""
        return BedrockAgent(
            identifier="test_agent",
            model=mock_llm,
            description="A test agent",
            memory=memory,
            system_message="You are a test assistant.",
        )

    @pytest.fixture
    def agent_with_tools(self, mock_llm, memory, mock_tool):
        """Create a BedrockAgent instance with tools."""
        mock_llm.response_type = "tool_call"
        return BedrockAgent(
            identifier="test_agent_with_tools",
            model=mock_llm,
            description="A test agent with tools",
            memory=memory,
            system_message="You are a test assistant with tools.",
            tools=[mock_tool],
        )

    # -------------------------------------------------------------------------
    # Tests
    # -------------------------------------------------------------------------

    def test_initialization(self, agent):
        """Test that the agent initializes correctly."""
        assert agent.identifier == "test_agent"
        assert agent.description == "A test agent"
        assert agent.memory.get_system_message() == "You are a test assistant."
        assert hasattr(agent, "tools")
        assert len(agent.tools) == 0
        assert "on_action" in agent.callbacks
        assert "on_observation" in agent.callbacks
        assert "on_answer" in agent.callbacks

    def test_initialization_with_tools(self, agent_with_tools, mock_tool):
        """Test that the agent initializes correctly with tools."""
        assert len(agent_with_tools.tools) == 1
        assert agent_with_tools.tools[0] == mock_tool

    def test_default_callbacks(self, agent):
        """Test the default callbacks."""
        # Test on_action
        with patch.object(agent, "log") as mock_log:
            agent._default_on_action("test_tool", {"param": "value"})
            mock_log.assert_called_once()
            assert (
                mock_log.call_args[0][0]
                == "Using tool: test_tool with parameters: {'param': 'value'}"
            )
            assert mock_log.call_args[0][1] == AgentLogType.ACTION

        # Test on_observation
        with patch.object(agent, "log") as mock_log:
            agent._default_on_observation("test_tool", "result")
            mock_log.assert_called_once()
            assert mock_log.call_args[0][0] == "Tool result: result"
            assert mock_log.call_args[0][1] == AgentLogType.OBSERVATION

        # Test on_answer
        with patch.object(agent, "log") as mock_log:
            agent._default_on_answer("answer text")
            mock_log.assert_called_once()
            assert mock_log.call_args[0][0] == "answer text"
            assert mock_log.call_args[0][1] == AgentLogType.ANSWER

    def test_register_tools(self, agent, mock_tool):
        """Test tool registration."""
        # Add a tool to the agent
        agent.tools = [mock_tool]

        # Mock the toolConfig attribute on the model
        agent.model.toolConfig = None

        # Call register_tools
        agent._register_tools()

        # Check that toolConfig was set correctly
        assert agent.model.toolConfig is not None
        assert hasattr(agent.model.toolConfig, "tools")
        assert len(agent.model.toolConfig.tools) == 1
        assert isinstance(agent.model.toolConfig.tools[0], BedrockTool)
        assert agent.model.toolConfig.tools[0].toolSpec.name == "test_tool"

    def test_register_tools_no_tools(self, agent):
        """Test that register_tools does nothing when there are no tools."""
        # Ensure the agent has no tools
        agent.tools = []

        # Call register_tools
        agent._register_tools()

        # Check that toolConfig was not set
        assert agent.model.toolConfig is None

    def test_process_tool_call(self, agent_with_tools, mock_tool):
        """Test processing a tool call."""
        # Set up a mock for the action callback
        agent_with_tools.callbacks["on_action"] = Mock()
        agent_with_tools.callbacks["on_observation"] = Mock()

        # Call process_tool_call
        result = agent_with_tools._process_tool_call("test_tool", {"param": "value"})

        # Check that the callbacks were called correctly
        agent_with_tools.callbacks["on_action"].assert_called_once_with(
            "test_tool", {"param": "value"}
        )
        agent_with_tools.callbacks["on_observation"].assert_called_once()

        # Check that the tool was called
        assert mock_tool.run_called
        assert mock_tool.run_args == {"environment": {}, "param": "value"}

        # Check the result format
        assert "Tool Result: test_tool" in result
        assert "Result from test_tool" in result

        # Check that the message was added to memory
        messages = agent_with_tools.memory.get_messages()
        assert len(messages) > 0
        assert messages[-1].role == "user"
        assert "SYSTEM DIRECTIVE" in messages[-1].content
        assert "test_tool returned: Result from test_tool" in messages[-1].content

    def test_process_tool_call_error(self, agent_with_tools):
        """Test processing a tool call that raises an error."""
        # Create a tool that raises an exception
        error_tool = MockTool(name="error_tool")
        error_tool.run = Mock(side_effect=Exception("Test error"))
        agent_with_tools.tools = [error_tool]

        # Set up mocks for the callbacks
        agent_with_tools.callbacks["on_action"] = Mock()
        agent_with_tools.callbacks["on_observation"] = Mock()

        # Call process_tool_call
        result = agent_with_tools._process_tool_call("error_tool", {"param": "value"})

        # Check that the callbacks were called correctly
        agent_with_tools.callbacks["on_action"].assert_called_once_with(
            "error_tool", {"param": "value"}
        )
        agent_with_tools.callbacks["on_observation"].assert_called_once_with(
            "error_tool", "Error: Test error"
        )

        # Check the result format
        assert "Tool Error: error_tool" in result
        assert "Test error" in result

        # Check that the error message was added to memory
        messages = agent_with_tools.memory.get_messages()
        assert len(messages) > 0
        assert messages[-1].role == "user"
        assert "Your tool call resulted in an error: Test error" in messages[-1].content

    def test_invoke_basic(self, agent, mock_llm):
        """Test the basic invoke functionality."""
        # Call invoke
        result = agent.invoke("Hello")

        # Check that the model's invoke method was called
        assert mock_llm.invoke_called

        # Check the result
        assert result == "This is a test response."

        # Check that the message was added to memory
        messages = agent.memory.get_messages()
        assert len(messages) == 2  # User prompt + assistant response
        assert messages[0].role == "user"
        assert messages[0].content == "Hello"
        assert messages[1].role == "assistant"
        assert messages[1].content == "This is a test response."

    def test_invoke_with_tool_call(self, agent_with_tools, mock_llm, mock_tool):
        """Test invoke with a tool call."""
        # Configure the mock LLM to return a tool call
        mock_llm.response_type = "tool_call"
        mock_llm.full_response = True

        # Mock the process_tool_call method
        with patch.object(agent_with_tools, "_process_tool_call") as mock_process:
            mock_process.return_value = (
                "[Tool Result: get_weather] Weather in New York: Sunny"
            )

            # Call invoke with max_iterations=1 to prevent multiple calls
            result = agent_with_tools.invoke(
                "What's the weather in New York?", max_iterations=1
            )

            # Check that process_tool_call was called correctly
            mock_process.assert_called_once_with(
                "get_weather", {"location": "New York"}
            )

            # Check the result
            assert "I'll help with that." in result
            assert "Weather in New York: Sunny" in result

    def test_stream_basic(self, agent, mock_llm):
        """Test the basic stream functionality."""
        # Call stream
        chunks = list(agent.stream("Hello"))

        # Check that the model's stream method was called
        assert mock_llm.stream_called

        # Check the chunks
        assert len(chunks) == 2
        assert chunks[0] == "This is a "
        assert chunks[1] == "test response."

        # Check that the message was added to memory
        messages = agent.memory.get_messages()
        assert len(messages) == 2  # User prompt + assistant response
        assert messages[0].role == "user"
        assert messages[0].content == "Hello"
        assert messages[1].role == "assistant"
        assert messages[1].content == "This is a test response."

    def test_stream_with_tool_call(self, agent_with_tools, mock_llm):
        """Test stream with a tool call."""
        # Configure the mock LLM to return a tool call
        mock_llm.response_type = "tool_call"
        mock_llm.full_response = True

        # Mock the process_tool_call method
        with patch.object(agent_with_tools, "_process_tool_call") as mock_process:
            mock_process.return_value = (
                "[Tool Result: get_weather] Weather in New York: Sunny"
            )

            # Call stream with max_iterations=1 to prevent multiple calls
            chunks = list(
                agent_with_tools.stream(
                    "What's the weather in New York?", max_iterations=1
                )
            )

            # Check that process_tool_call was called correctly
            mock_process.assert_called_once()
            assert mock_process.call_args[0][0] == "get_weather"

            # Check the chunks
            assert len(chunks) == 1
            assert chunks[0] == "I'll help with that."

    def test_run_with_stream_true(self, agent):
        """Test the run method with stream=True."""
        with patch.object(agent, "stream") as mock_stream:
            mock_stream.return_value = iter(["This is a ", "test response."])

            # Call run with stream=True
            result = agent.run("Hello", stream=True)

            # Check that stream was called correctly
            mock_stream.assert_called_once_with("Hello", 10)

            # Check that the result is an iterator
            assert hasattr(result, "__iter__")

    def test_run_with_stream_false(self, agent):
        """Test the run method with stream=False."""
        with patch.object(agent, "invoke") as mock_invoke:
            mock_invoke.return_value = "This is a test response."

            # Call run with stream=False
            result = agent.run("Hello", stream=False)

            # Check that invoke was called correctly
            mock_invoke.assert_called_once_with("Hello", 10)

            # Check that the result is a string
            assert isinstance(result, str)
            assert result == "This is a test response."

    def test_memory_flushing(self, agent):
        """Test that memory is properly flushed between runs."""
        # Add some messages to memory
        agent.memory.add_message(role="user", content="Previous message")
        agent.memory.add_message(role="assistant", content="Previous response")

        # Verify the messages are there
        assert len(agent.memory.get_messages()) == 2

        # Run the agent again - should flush the memory
        agent.run("New message")

        # Check that only the new message and response are in memory
        messages = agent.memory.get_messages()
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[0].content == "New message"
        assert messages[1].role == "assistant"

    def test_multiple_iterations(self, agent_with_tools, mock_llm):
        """Test multiple iterations with tool calls."""
        # Configure the mock LLM to return a tool call
        mock_llm.response_type = "tool_call"
        mock_llm.full_response = True

        # Make invoke return different responses on subsequent calls
        original_invoke = mock_llm.invoke
        invoke_count = 0

        def mock_invoke(prompt, **kwargs):
            nonlocal invoke_count
            if invoke_count == 0:
                invoke_count += 1
                return original_invoke(prompt, **kwargs)
            else:
                # Second call should be a normal response without tool calls
                return {
                    "output": {
                        "message": {
                            "content": [{"text": "Final answer after tool call."}]
                        }
                    },
                    "stopReason": "end_turn",
                }

        mock_llm.invoke = mock_invoke

        # Call invoke
        with patch.object(agent_with_tools, "_process_tool_call") as mock_process:
            mock_process.return_value = (
                "[Tool Result: get_weather] Weather in New York: Sunny"
            )

            result = agent_with_tools.invoke("What's the weather in New York?")

            # Check that process_tool_call was called exactly once
            mock_process.assert_called_once()

            # Check the result includes both the initial and final responses
            assert "I'll help with that." in result
            assert "Weather in New York: Sunny" in result
            assert "Final answer after tool call." in result

    def test_max_iterations(self, agent_with_tools, mock_llm):
        """Test that the agent respects max_iterations."""
        # Configure the mock LLM to always return a tool call
        mock_llm.response_type = "tool_call"
        mock_llm.full_response = True

        # Patch _process_tool_call to avoid actual tool execution
        with patch.object(agent_with_tools, "_process_tool_call") as mock_process:
            mock_process.return_value = "[Tool Result: get_weather] Weather data"

            # Call invoke with max_iterations=2
            with patch.object(
                agent_with_tools.model, "invoke", wraps=mock_llm.invoke
            ) as mock_model_invoke:
                agent_with_tools.invoke("What's the weather?", max_iterations=2)

                # Check that the model's invoke method was called exactly 2 times
                assert mock_model_invoke.call_count == 2


if __name__ == "__main__":
    pytest.main(["-xvs", "test_bedrock_agent.py"])
