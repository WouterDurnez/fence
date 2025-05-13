import pytest

from fence.memory.base import FleetingMemory


@pytest.fixture
def memory():
    """
    Fixture to create a new instance of FleetingMemory for each test.
    """
    return FleetingMemory()


def test_add_system_message(memory):
    """
    Test adding and retrieving a system message.
    """
    system_message = "This is a system message"
    memory.set_system_message(system_message)
    assert memory.get_system_message() == system_message


def test_add_user_message(memory):
    """
    Test adding and retrieving a user message.
    """
    user_message = "This is a user message"
    memory.add_user_message(user_message)
    messages = memory.get_messages()
    assert len(messages) == 1
    assert messages[0].role == "user"
    assert messages[0].content[0].text == user_message


def test_add_assistant_message(memory):
    """
    Test adding and retrieving an assistant message.
    """
    assistant_message = "This is an assistant message"
    memory.add_assistant_message(assistant_message)
    messages = memory.get_messages()
    assert len(messages) == 1
    assert messages[0].role == "assistant"
    assert messages[0].content[0].text == assistant_message


def test_add_multiple_messages(memory):
    """
    Test adding multiple messages to memory.
    """
    user_message = "User message"
    assistant_message = "Assistant message"
    memory.add_user_message(user_message)
    memory.add_assistant_message(assistant_message)
    messages = memory.get_messages()

    assert len(messages) == 2
    assert messages[0].role == "user"
    assert messages[0].content[0].text == user_message
    assert messages[1].role == "assistant"
    assert messages[1].content[0].text == assistant_message


def test_invalid_role_raises_error(memory):
    """
    Test that adding a message with an invalid role raises a ValueError.
    """
    with pytest.raises(
        ValueError, match="Role must be 'system', 'user', or 'assistant'"
    ):
        memory.add_message(role="invalid", content="Invalid role message")
