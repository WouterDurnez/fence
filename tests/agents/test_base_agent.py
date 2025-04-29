"""
Tests for the BaseAgent class.

This module contains tests for the BaseAgent implementation, including:
1. Initialization with different configurations
2. Memory management and flushing
3. Logging functionality
4. TOML formatting
"""

import os
from unittest.mock import patch

import pytest

from fence.agents.base import AgentLogType, BaseAgent
from fence.memory.base import FleetingMemory
from fence.models.base import LLM

# Check if OpenAI API key is present for tests that might use real LLMs
has_openai_api_key = os.environ.get("OPENAI_API_KEY") is not None


class BaseAgentMock(BaseAgent):
    """Implementation of BaseAgent for testing."""

    def run(self, prompt: str) -> str:
        """Implement abstract method for testing."""
        return f"Response to: {prompt}"


class MockLLM(LLM):
    """Mock LLM for testing."""

    def __init__(self, **kwargs):
        """Initialize mock LLM."""
        super().__init__(**kwargs)
        self.model_name = "mock_llm"

    def invoke(self, prompt, **kwargs):
        """Mock implementation of invoke."""
        return f"Mock response to: {prompt}"


class TestBaseAgent:
    """Tests for the BaseAgent class."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM instance."""
        return MockLLM(source="test")

    @pytest.fixture
    def memory(self):
        """Create a memory instance."""
        memory = FleetingMemory()
        memory.system_message = "You are a test agent"
        return memory

    @pytest.fixture
    def base_agent(self, mock_llm, memory):
        """Create a base agent instance."""
        return BaseAgentMock(
            identifier="test_agent",
            model=mock_llm,
            description="Test agent description",
            memory=memory,
            environment={"test_key": "test_value"},
            prefill="Test prefill",
        )

    def test_initialization(self, base_agent, mock_llm, memory):
        """Test that BaseAgent initializes correctly."""
        assert base_agent.identifier == "test_agent"
        assert base_agent.model == mock_llm
        assert base_agent.description == "Test agent description"
        assert base_agent.memory == memory
        assert base_agent.environment == {"test_key": "test_value"}
        assert base_agent.memory.system_message == "You are a test agent"

    def test_initialization_defaults(self):
        """Test BaseAgent initialization with default values."""
        agent = BaseAgentMock()
        assert agent.identifier == "BaseAgentMock"
        assert agent.model is None
        assert agent.description is None
        assert agent.memory is not None
        assert agent.environment == {}
        assert agent.log_agentic_response is True

    def test_run_method(self, base_agent):
        """Test the run method implementation."""
        response = base_agent.run("test prompt")
        assert response == "Response to: test prompt"

    def test_format_toml(self, base_agent):
        """Test the format_toml method."""
        toml_string = base_agent.format_toml()
        assert "[[agents]]" in toml_string
        assert 'agent_name = """test_agent"""' in toml_string
        assert 'agent_description = """Test agent description"""' in toml_string

    def test_log_with_string_type(self, base_agent):
        """Test the log method with string type."""
        # Test that logging works with string type
        with patch("builtins.print") as mock_print:
            base_agent.log("Test message", "thought")
            mock_print.assert_called_once()

    def test_log_with_enum_type(self, base_agent):
        """Test the log method with enum type."""
        # Test that logging works with enum type
        with patch("builtins.print") as mock_print:
            base_agent.log("Test message", AgentLogType.THOUGHT)
            mock_print.assert_called_once()

    def test_log_with_disabled_logging(self):
        """Test log method with logging disabled."""
        agent = BaseAgentMock(log_agentic_response=False)
        with patch("builtins.print") as mock_print:
            agent.log("Test message", AgentLogType.THOUGHT)
            mock_print.assert_not_called()
