"""
Tests for AgentCore memory implementation.
"""

from unittest.mock import MagicMock, patch

import pytest

from fence.memory.agentcore import AgentCoreMemory
from fence.templates.models import Content, TextContent

############
# Fixtures #
############


@pytest.fixture
def mock_client():
    """Mock AgentCore client."""
    client = MagicMock()
    client.create_event.return_value = {"eventId": "test-event-id"}
    client.list_events.return_value = []
    return client


@pytest.fixture
def mock_agentcore_memory():
    """Mock AgentCore memory object."""
    return {"id": "test-memory-id", "name": "test-memory"}


@pytest.fixture
def agentcore_memory(mock_client, mock_agentcore_memory):
    """Create AgentCoreMemory instance for testing."""
    return AgentCoreMemory(
        agentcore_memory=mock_agentcore_memory,
        client=mock_client,
        actor_id="test-actor",
        session_id="test-session",
    )


@pytest.fixture
def sample_events():
    """Sample events from AgentCore."""
    return [
        {
            "eventId": "event-1",
            "eventTimestamp": "2024-01-01T10:00:00Z",
            "payload": [
                {
                    "conversational": {
                        "role": "USER",
                        "content": {"text": "Hello, how are you?"},
                    }
                }
            ],
        },
        {
            "eventId": "event-2",
            "eventTimestamp": "2024-01-01T10:01:00Z",
            "payload": [
                {
                    "conversational": {
                        "role": "ASSISTANT",
                        "content": {"text": "I'm doing well, thank you!"},
                    }
                }
            ],
        },
        {
            "eventId": "event-3",
            "eventTimestamp": "2024-01-01T10:02:00Z",
            "payload": [
                {
                    "conversational": {
                        "role": "USER",
                        "content": {"text": "What can you help me with?"},
                    }
                }
            ],
        },
    ]


#########
# Tests #
#########


def test_initialization(mock_client, mock_agentcore_memory):
    """Test AgentCoreMemory initialization."""
    memory = AgentCoreMemory(
        agentcore_memory=mock_agentcore_memory,
        client=mock_client,
        actor_id="test-actor",
        session_id="test-session",
    )

    assert memory.agentcore_memory == mock_agentcore_memory
    assert memory.client == mock_client
    assert memory.actor_id == "test-actor"
    assert memory.session_id == "test-session"
    assert memory.system is None
    assert memory.messages == []


def test_add_message_string_content(agentcore_memory, mock_client):
    """Test adding a message with string content."""
    agentcore_memory.add_message("user", "Hello world")

    # Verify the client was called correctly
    mock_client.create_event.assert_called_once_with(
        memory_id="test-memory-id",
        actor_id="test-actor",
        session_id="test-session",
        messages=[("Hello world", "USER")],
    )


def test_add_message_text_content(agentcore_memory, mock_client):
    """Test adding a message with TextContent object."""
    content = TextContent(text="Hello world")
    agentcore_memory.add_message("assistant", content)

    # Verify the client was called correctly
    mock_client.create_event.assert_called_once_with(
        memory_id="test-memory-id",
        actor_id="test-actor",
        session_id="test-session",
        messages=[("Hello world", "ASSISTANT")],
    )


def test_add_system_message(agentcore_memory, mock_client):
    """Test adding a system message."""
    agentcore_memory.add_message("system", "You are a helpful assistant")

    # System messages should not trigger create_event
    mock_client.create_event.assert_not_called()

    # System message should be stored locally
    assert isinstance(agentcore_memory.system, TextContent)
    assert agentcore_memory.system.text == "You are a helpful assistant"


def test_set_system_message(agentcore_memory):
    """Test set_system_message method."""
    agentcore_memory.set_system_message("You are a helpful assistant")

    assert isinstance(agentcore_memory.system, TextContent)
    assert agentcore_memory.system.text == "You are a helpful assistant"


def test_get_system_message_text_content(agentcore_memory):
    """Test getting system message when it's TextContent."""
    agentcore_memory.system = TextContent(text="System message")

    result = agentcore_memory.get_system_message()
    assert result == "System message"


def test_get_system_message_string(agentcore_memory):
    """Test getting system message when it's a string."""
    agentcore_memory.system = "System message"

    result = agentcore_memory.get_system_message()
    assert result == "System message"


def test_get_system_message_none(agentcore_memory):
    """Test getting system message when it's None."""
    result = agentcore_memory.get_system_message()
    assert result is None


def test_add_message_invalid_role(agentcore_memory):
    """Test adding a message with invalid role raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported role: invalid"):
        agentcore_memory.add_message("invalid", "Some content")


def test_add_message_unsupported_content_type(agentcore_memory):
    """Test adding a message with unsupported content type raises TypeError."""
    # Create a mock Content object that's not TextContent
    mock_content = MagicMock(spec=Content)

    with pytest.raises(
        TypeError, match="Only TextContent is supported in AgentCoreMemory"
    ):
        agentcore_memory.add_message("user", mock_content)


def test_get_messages(agentcore_memory, mock_client, sample_events):
    """Test getting messages from AgentCore."""
    mock_client.list_events.return_value = sample_events

    messages = agentcore_memory.get_messages()

    # Verify client was called correctly
    mock_client.list_events.assert_called_once_with(
        memory_id="test-memory-id", actor_id="test-actor", session_id="test-session"
    )

    # Verify messages structure
    assert len(messages) == 3
    assert messages[0].role == "user"
    assert messages[0].content[0].text == "Hello, how are you?"
    assert messages[1].role == "assistant"
    assert messages[1].content[0].text == "I'm doing well, thank you!"
    assert messages[2].role == "user"
    assert messages[2].content[0].text == "What can you help me with?"


def test_get_messages_missing_requirements(agentcore_memory):
    """Test getting messages when required fields are missing."""
    # Test with missing agentcore_memory
    agentcore_memory.agentcore_memory = None
    with pytest.raises(
        ValueError, match="Memory, actor_id, and session_id must be set"
    ):
        agentcore_memory.get_messages()

    # Reset and test with missing actor_id
    agentcore_memory.agentcore_memory = {"id": "test-memory-id"}
    agentcore_memory.actor_id = None
    with pytest.raises(
        ValueError, match="Memory, actor_id, and session_id must be set"
    ):
        agentcore_memory.get_messages()

    # Reset and test with missing session_id
    agentcore_memory.actor_id = "test-actor"
    agentcore_memory.session_id = None
    with pytest.raises(
        ValueError, match="Memory, actor_id, and session_id must be set"
    ):
        agentcore_memory.get_messages()


def test_fetch_recent_messages(agentcore_memory, mock_client, sample_events):
    """Test fetching recent messages with limit."""
    mock_client.list_events.return_value = sample_events[:2]

    result = agentcore_memory._fetch_recent_messages(max_results=2)

    # Verify client was called with correct parameters
    mock_client.list_events.assert_called_once_with(
        memory_id="test-memory-id",
        actor_id="test-actor",
        session_id="test-session",
        max_results=2,
    )

    assert len(result) == 2


def test_events_sorting(agentcore_memory, mock_client):
    """Test that events are sorted by timestamp."""
    # Events in wrong chronological order
    unsorted_events = [
        {
            "eventId": "event-2",
            "eventTimestamp": "2024-01-01T10:01:00Z",
            "payload": [
                {
                    "conversational": {
                        "role": "ASSISTANT",
                        "content": {"text": "Second message"},
                    }
                }
            ],
        },
        {
            "eventId": "event-1",
            "eventTimestamp": "2024-01-01T10:00:00Z",
            "payload": [
                {
                    "conversational": {
                        "role": "USER",
                        "content": {"text": "First message"},
                    }
                }
            ],
        },
    ]

    mock_client.list_events.return_value = unsorted_events

    messages = agentcore_memory.get_messages()

    # Verify messages are in correct chronological order
    assert messages[0].content[0].text == "First message"
    assert messages[1].content[0].text == "Second message"


@patch("builtins.print")
def test_get_messages_debug_print(mock_print, agentcore_memory, mock_client):
    """Test that get_messages prints debug information."""
    mock_client.list_events.return_value = []

    agentcore_memory.get_messages()

    # Verify debug print was called
    mock_print.assert_called_once_with("requested all messages: 0")
