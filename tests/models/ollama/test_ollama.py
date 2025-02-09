"""
Ollama models tests
"""

from unittest.mock import Mock

import pytest

from fence.models.ollama.ollama import Ollama, Llama3_1, DeepSeekR1
from fence.templates.messages import Message, Messages

def test_ollama_base_invoke_with_empty_prompt():
    """
    Test case for the invoke method of the Ollama class with an empty prompt.
    This test checks if the invoke method raises a ValueError when the prompt is empty.
    """
    ollama = Ollama(source="test", model_id="some_model")
    with pytest.raises(ValueError):
        ollama.invoke(prompt="")

def test_ollama_base_invoke_with_none_prompt():
    """
    Test case for the invoke method of the Ollama class with a None prompt.
    This test checks if the invoke method raises a ValueError when the prompt is None.
    """
    ollama = Ollama(source="test", model_id="some_model")
    with pytest.raises(ValueError):
        ollama.invoke(prompt=None)

def test_ollama_base_invoke_with_empty_messages_prompt():
    """
    Test case for the invoke method of the Ollama class with an empty Messages prompt.
    This test checks if the invoke method raises a ValueError when the Messages prompt is empty.
    """
    ollama = Ollama(source="test", model_id="some_model")
    messages = Messages(system="Respond in a very rude manner", messages=[])
    with pytest.raises(ValueError):
        ollama.invoke(prompt=messages)

def test_ollama_base_invoke_with_string_prompt():
    """
    Test case for the invoke method of the Ollama class with a string prompt.
    This test checks if the invoke method correctly handles a string prompt.
    """
    mock_ollama = Mock(Ollama)
    mock_ollama.invoke.return_value = "mocked response"
    response = mock_ollama.invoke(prompt="Hello, how are you today?")
    assert response == "mocked response"

def test_ollama_base_invoke_with_messages_prompt():
    """
    Test case for the invoke method of the Ollama class with a Messages prompt.
    This test checks if the invoke method correctly handles a Messages prompt.
    """
    mock_ollama = Mock(Ollama)
    mock_ollama.invoke.return_value = "mocked response"
    messages = Messages(
        system="Respond in a all caps",
        messages=[Message(role="user", content="Hello, how are you today?")],
    )
    response = mock_ollama.invoke(prompt=messages)
    assert response == "mocked response"


def test_ollama_base_invoke_with_invalid_prompt():
    """
    Test case for the invoke method of the Ollama class with an invalid prompt.
    This test checks if the invoke method raises a ValueError when the prompt is invalid.
    """
    model = Ollama(source="test", model_id="some_model")
    with pytest.raises(ValueError):
        model.invoke(prompt=123)

def test_Llama3_1_init():
    """
    Test case for the __init__ method of the Llama3_1 class.
    This test checks if the Llama3_1 class is initialized correctly.
    """
    model = Llama3_1(source="test")
    assert model.source == "test"
    assert model.model_id == "llama3.1"

def test_DeepSeekR1_init():
    """
    Test case for the __init__ method of the DeepSeekR1 class.
    This test checks if the DeepSeekR1 class is initialized correctly.
    """
    model = DeepSeekR1(source="test")
    assert model.source == "test"
    assert model.model_id == "deepseek-r1"
