"""
Message template tests
"""

import logging

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


@pytest.fixture
def nested_messages_template():
    """
    This fixture creates a MessagesTemplate object for testing purposes.
    """
    messages = Messages(
        system="System message {system_var}",
        messages=[
            Message(
                role="user",
                content=[TextContent(text="User message {user_var.nested_var}")],
            ),
            Message(
                role="assistant", content="Assistant message {assistant_var.nested_var}"
            ),
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


def test_messages_template_render_missing_variable(messages_template, caplog):
    """
    Test case for the render method of the MessagesTemplate class when a required variable is missing.
    This test checks if the render method logs a warning when a required variable is missing.
    """
    # Assuming messages_template.input_variables contains 'system_var' and 'user_var'
    input_dict = {"system_var": "test1"}  # Missing 'user_var'

    with caplog.at_level(logging.WARNING):
        _ = messages_template.render(input_dict=input_dict)

    # Check if the warning about missing variables was logged
    assert any(
        "Possible missing variables" in message for message in caplog.text.splitlines()
    )


def test_messages_template_render_superfluous_variable(messages_template, caplog):
    """
    Test case for the render method of the MessagesTemplate class when a superfluous variable is provided.
    This test checks if the render method logs a debug message when a superfluous variable is provided.
    """
    # Assuming messages_template.input_variables contains 'system_var' and 'user_var'
    input_dict = {
        "system_var": "test1",
        "user_var": "test2",
        "assistant_var": "test3",
        "superfluous_var": "test4",
    }

    with caplog.at_level(logging.DEBUG):
        _ = messages_template.render(input_dict=input_dict)

    # Check if the debug message about superfluous variables was logged
    assert any(
        "Superfluous variables" in message for message in caplog.text.splitlines()
    )


def test_messages_template_nested_placeholder(nested_messages_template):
    """
    Test case for the MessagesTemplate class with nested
    placeholders in the source.
    """

    assert set(nested_messages_template.input_variables) == {
        "system_var",
        "user_var.nested_var",
        "assistant_var.nested_var",
    }
    input_dict = {
        "system_var": "test1",
        "user_var": {"nested_var": "test2"},
        "assistant_var": {"nested_var": "test3"},
    }
    rendered_messages = nested_messages_template.render(input_dict=input_dict)
    assert rendered_messages.system == "System message test1"
    assert rendered_messages.messages[0].content[0].text == "User message test2"
    assert rendered_messages.messages[1].content == "Assistant message test3"


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
