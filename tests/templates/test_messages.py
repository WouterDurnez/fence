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
def messages_template_no_array():
    """
    This fixture creates a MessagesTemplate object for testing purposes.
    """
    messages = Messages(
        system="System message {system_var}",
        messages=[
            Message(role="user", content="User message {user_var}"),
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


def test_model_dump_gemini(messages_template):
    """
    Test case for the model_dump_gemini method of the MessagesTemplate class.
    This test checks if the model_dump_gemini method correctly converts the MessagesTemplate instance to a dict.
    """
    input_dict = {"system_var": "test1", "user_var": "test2", "assistant_var": "test3"}
    rendered_messages = messages_template.render(input_dict=input_dict)
    gemini = rendered_messages.model_dump_gemini()
    assert gemini == {
        'contents': [{'parts': [{'text': 'User message test2'}], 'role': 'user'}, {'parts': {'text': 'Assistant message test3'}, 'role': 'model'}],
        'system_instruction': {'parts': {'text': 'System message test1'}}
        }
def test_model_dump_bedrock_converse(messages_template):
    """
    Test case for the dump_bedrock_converse method of the MessagesTemplate class.
    This test checks if the dump_bedrock_converse method correctly converts the MessagesTemplate instance to a dict.
    """
    input_dict = {"system_var": "test1", "user_var": "test2", "assistant_var": "test3"}
    rendered_messages = messages_template.render(input_dict=input_dict)
    bedrock_converse = rendered_messages.model_dump_bedrock_converse()
    assert bedrock_converse == {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "text": "User message test2"
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "text": "Assistant message test3"
                    }
                ]
            }
        ],
        "system": [
            {
                "text": "System message test1"
            }
        ]
    }

def test_model_dump_openai(messages_template):
    """
    Test case for the dump_openai method of the MessagesTemplate class.
    This test checks if the dump_openai method correctly converts the MessagesTemplate instance to a dict.
    """
    input_dict = {"system_var": "test1", "user_var": "test2", "assistant_var": "test3"}
    rendered_messages = messages_template.render(input_dict=input_dict)
    openai = rendered_messages.model_dump_openai()
    assert openai == [
        {
            'role': 'user',
            'content': [{'text': 'User message test2', 'type': 'text'}]
        },
        {
            'role': 'assistant',
            'content': 'Assistant message test3'
        },
        {
            'content': 'System message test1', 
            'role': 'system'
        }
    ]

def test_model_dump_ollama_with_array(messages_template):
    """
    Test case for the dump_ollama method of the MessagesTemplate class.
    This test checks if the dump_ollama raises a TypeError when the content is an array.
    """
    input_dict = {"system_var": "test1", "user_var": "test2", "assistant_var": "test3"}
    rendered_messages = messages_template.render(input_dict=input_dict)
    with pytest.raises(TypeError):
        rendered_messages.model_dump_ollama()

def test_model_dump_ollama_no_array(messages_template_no_array):
    """
    Test case for the dump_ollama method of the MessagesTemplate class.
    This test checks if the dump_ollama method correctly converts the MessagesTemplate instance to a dict.
    """
    input_dict = {"system_var": "test1", "user_var": "test2", "assistant_var": "test3"}
    rendered_messages = messages_template_no_array.render(input_dict=input_dict)
    ollama = rendered_messages.model_dump_ollama()
    assert ollama == [
        {'role': 'system', 'content': 'System message test1'},
        {'role': 'user', 'content': 'User message test2'},
        {'role': 'assistant', 'content': 'Assistant message test3'}]


def test_model_dump_mistral(messages_template):
    """
    Test case for the model_dump_mistral method of the MessagesTemplate class.
    This test checks if the model_dump_mistral method correctly converts the MessagesTemplate instance to a dict.
    """
    input_dict = {"system_var": "test1", "user_var": "test2", "assistant_var": "test3"}
    rendered_messages = messages_template.render(input_dict=input_dict)
    with pytest.raises(TypeError):
        rendered_messages.model_dump_mistral()

def test_model_dump_mistral_no_array(messages_template_no_array):
    """
    Test case for the model_dump_mistral method of the MessagesTemplate class.
    This test checks if the model_dump_mistral method correctly converts the MessagesTemplate instance to a dict.
    """
    input_dict = {"system_var": "test1", "user_var": "test2", "assistant_var": "test3"}
    rendered_messages = messages_template_no_array.render(input_dict=input_dict)
    mistral = rendered_messages.model_dump_mistral()
    assert mistral == [
        {'role': 'system', 'content': 'System message test1'},
        {'role': 'user', 'content': 'User message test2'},
        {'role': 'assistant', 'content': 'Assistant message test3'}]

