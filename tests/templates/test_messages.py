import pytest

from fence.templates.messages import MessagesTemplate
from fence.templates.models import Message, Messages, TextContent


@pytest.fixture
def messages_template():
    """
    This fixture creates a MessagesTemplate object for testing purposes.
    """
    messages = Messages(
        system="System message {system_var}",
        messages=[
            Message(role="user", content=[TextContent(text="User message {user_var}")]),
            Message(role="assistant", content="Assistant message {assistant_var}"),
        ],
    )
    return MessagesTemplate(source=messages)


def test_messages_template_initialization(messages_template):
    """
    Test case for the initialization of the MessagesTemplate class.
    This test checks if the input variables are correctly identified during initialization.
    """
    assert set(messages_template.input_variables) == {
        "system_var",
        "user_var",
        "assistant_var",
    }


def test_messages_template_render(messages_template):
    """
    Test case for the render method of the MessagesTemplate class.
    This test checks if the render method correctly replaces the placeholders with the provided variables.
    """
    input_dict = {"system_var": "test1", "user_var": "test2", "assistant_var": "test3"}
    rendered_messages = messages_template.render(input_dict=input_dict)
    assert rendered_messages.system == "System message test1"
    assert rendered_messages.messages[0].content[0].text == "User message test2"
    assert rendered_messages.messages[1].content == "Assistant message test3"


def test_messages_template_render_missing_variable(messages_template):
    """
    Test case for the render method of the MessagesTemplate class when a required variable is missing.
    This test checks if the render method raises a ValueError when a required variable is missing.
    """
    input_dict = {"system_var": "test1", "user_var": "test2"}
    with pytest.raises(ValueError):
        messages_template.render(input_dict=input_dict)


def test_messages_template_add(messages_template):
    """
    Test case for the __add__ method of the MessagesTemplate class.
    This test checks if the __add__ method correctly combines two MessagesTemplate instances.
    """
    other_messages = Messages(
        system="Other system message {other_system_var}",
        messages=[
            Message(
                role="user",
                content=[TextContent(text="Other user message {other_user_var}")],
            ),
            Message(
                role="assistant",
                content="Other assistant message {other_assistant_var}",
            ),
        ],
    )
    other_messages_template = MessagesTemplate(source=other_messages)
    combined_messages_template = messages_template + other_messages_template
    combined_input_vars = set(combined_messages_template.input_variables)
    expected_input_vars = {
        "system_var",
        "user_var",
        "assistant_var",
        "other_system_var",
        "other_user_var",
        "other_assistant_var",
    }
    assert combined_input_vars == expected_input_vars


def test_messages_template_eq(messages_template):
    """
    Test case for the __eq__ method of the MessagesTemplate class.
    This test checks if the __eq__ method correctly identifies when two MessagesTemplate instances are equal.
    """
    other_messages_template = MessagesTemplate(source=messages_template.source)
    assert messages_template == other_messages_template
