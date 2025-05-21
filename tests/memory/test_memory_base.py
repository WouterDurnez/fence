import pytest

from fence.memory.base import FleetingMemory
from fence.models.base import LLM
from fence.templates.models import Messages


# Mock LLM for testing
class MockLLM(LLM):
    """Mock LLM that returns a predefined response."""

    def __init__(self, response="Test response"):
        """
        Initialize with a predefined response.
        :param response: The response to return when invoked.
        """
        super().__init__()
        self.model_id = "mock-model"
        self.model_name = "Mock Model"
        self.response = response

    def invoke(self, prompt: str | Messages, **kwargs) -> str:
        """
        Return the predefined response.
        :param prompt: The prompt (ignored in this mock).
        :param kwargs: Additional parameters (ignored in this mock).
        :return: The predefined response.
        """
        return self.response


@pytest.fixture
def memory():
    """
    Fixture to create a new instance of FleetingMemory for each test.
    """
    return FleetingMemory()


@pytest.fixture
def populated_memory():
    """
    Fixture to create a memory instance with populated messages.
    """
    memory = FleetingMemory()
    memory.set_system_message("Test system message")
    memory.add_user_message("User message 1")
    memory.add_assistant_message("Assistant message 1")
    memory.add_user_message("User message 2")
    memory.add_assistant_message("Assistant message 2")
    return memory


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


def test_generate_summary(populated_memory):
    """
    Test generating a summary of the memory.
    """
    expected_summary = "This is a summary of the conversation."
    model = MockLLM(response=expected_summary)

    # Default n_messages
    summary = populated_memory.generate_summary(model)
    assert summary == expected_summary

    # Custom n_messages
    summary = populated_memory.generate_summary(model, n_messages=2)
    assert summary == expected_summary


def test_generate_title(populated_memory):
    """
    Test generating a title for the memory.
    """
    expected_title = "Test Title"
    model = MockLLM(response=expected_title)

    # Default n_messages (None)
    title = populated_memory.generate_title(model)
    assert title == expected_title

    # Positive n_messages (first n messages)
    title = populated_memory.generate_title(model, n_messages=2)
    assert title == expected_title

    # Negative n_messages (last n messages)
    title = populated_memory.generate_title(model, n_messages=-2)
    assert title == expected_title

    # Zero n_messages (should use all messages)
    title = populated_memory.generate_title(model, n_messages=0)
    assert title == expected_title
