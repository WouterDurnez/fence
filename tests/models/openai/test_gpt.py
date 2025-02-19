"""
OpenAI GPT model tests
"""

from unittest.mock import Mock

import pytest

from fence.models.openai.gpt import GPT4o
from fence.templates.messages import Message, Messages


def test_gpt_base_invoke_with_empty_prompt():
    """
    Test case for the invoke method of the GPTBase class with an empty prompt.
    This test checks if the invoke method raises a ValueError when the prompt is empty.
    """
    gpt = GPT4o(source="test")
    with pytest.raises(ValueError):
        gpt.invoke(prompt="")


def test_gpt_base_invoke_with_none_prompt():
    """
    Test case for the invoke method of the GPTBase class with a None prompt.
    This test checks if the invoke method raises a ValueError when the prompt is None.
    """
    gpt = GPT4o(source="test")
    with pytest.raises(ValueError):
        gpt.invoke(prompt=None)


def test_gpt_base_invoke_with_empty_messages_prompt():
    """
    Test case for the invoke method of the GPTBase class with an empty Messages prompt.
    This test checks if the invoke method raises a ValueError when the Messages prompt is empty.
    """
    gpt = GPT4o(source="test")
    messages = Messages(system="Respond in a very rude manner", messages=[])
    with pytest.raises(ValueError):
        gpt.invoke(prompt=messages)


def test_gpt_base_invoke_with_string_prompt():
    """
    Test case for the invoke method of the GPTBase class with a string prompt.
    This test checks if the invoke method correctly handles a string prompt.
    """
    mock_llm = Mock(GPT4o)
    mock_llm.invoke.return_value = "mocked response"
    response = mock_llm.invoke(prompt="Hello, how are you today?")
    assert response == "mocked response"


def test_gpt_base_invoke_with_messages_prompt():
    """
    Test case for the invoke method of the GPTBase class with a Messages prompt.
    This test checks if the invoke method correctly handles a Messages prompt.
    """
    mock_llm = Mock(GPT4o)
    mock_llm.invoke.return_value = "mocked response"
    messages = Messages(
        system="Respond in a very rude manner",
        messages=[Message(role="user", content="Hello, how are you today?")],
    )
    response = mock_llm.invoke(prompt=messages)
    assert response == "mocked response"


def test_gpt_base_invoke_with_invalid_prompt():
    """
    Test case for the invoke method of the GPTBase class with an invalid prompt.
    This test checks if the invoke method raises a ValueError when the prompt is not a string or a Messages object.
    """
    gpt = GPT4o(source="test")
    with pytest.raises(ValueError):
        gpt.invoke(prompt=123)


def test_gpt4o_init():
    """
    Test case for the __init__ method of the GPT4o class.
    This test checks if the __init__ method correctly initializes a GPT4o object.
    """
    gpt = GPT4o(source="test")
    assert gpt.source == "test"
    assert gpt.model_id == "gpt-4o"
    assert gpt.model_name == "GPT 4o"
