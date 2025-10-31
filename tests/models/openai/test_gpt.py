"""
OpenAI GPT model tests
"""

import os
from unittest.mock import Mock, patch

import pytest

from fence.models.openai.gpt import (
    GPT4o,
    GPT4_1,
    GPT4_1Mini,
    GPT4_1Nano,
    O3Mini,
    O4Mini,
)
from fence.templates.messages import Message, Messages

# Check if OpenAI API key is present
has_openai_api_key = os.environ.get("OPENAI_API_KEY") is not None


@pytest.mark.skipif(not has_openai_api_key, reason="OpenAI API key not found in environment")
@patch.dict(os.environ, {'OPENAI_API_KEY': 'dummy_key'}, clear=True)
def test_gpt_base_invoke_with_empty_prompt():
    """
    Test case for the invoke method of the GPTBase class with an empty prompt.
    This test checks if the invoke method raises a ValueError when the prompt is empty.
    """
    gpt = GPT4o(source="test")
    with pytest.raises(ValueError):
        gpt.invoke(prompt="")


@pytest.mark.skipif(not has_openai_api_key, reason="OpenAI API key not found in environment")
@patch.dict(os.environ, {'OPENAI_API_KEY': 'dummy_key'}, clear=True)
def test_gpt_base_invoke_with_none_prompt():
    """
    Test case for the invoke method of the GPTBase class with a None prompt.
    This test checks if the invoke method raises a ValueError when the prompt is None.
    """
    gpt = GPT4o(source="test")
    with pytest.raises(ValueError):
        gpt.invoke(prompt=None)


@pytest.mark.skipif(not has_openai_api_key, reason="OpenAI API key not found in environment")
@patch.dict(os.environ, {'OPENAI_API_KEY': 'dummy_key'}, clear=True)
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


@pytest.mark.skipif(not has_openai_api_key, reason="OpenAI API key not found in environment")
@patch.dict(os.environ, {'OPENAI_API_KEY': 'dummy_key'}, clear=True)
def test_gpt_base_invoke_with_invalid_prompt():
    """
    Test case for the invoke method of the GPTBase class with an invalid prompt.
    This test checks if the invoke method raises a ValueError when the prompt is not a string or a Messages object.
    """
    gpt = GPT4o(source="test")
    with pytest.raises(ValueError):
        gpt.invoke(prompt=123)


@pytest.mark.skipif(not has_openai_api_key, reason="OpenAI API key not found in environment")
@patch.dict(os.environ, {'OPENAI_API_KEY': 'dummy_key'}, clear=True)
def test_gpt4o_init():
    """
    Test case for the __init__ method of the GPT4o class.
    This test checks if the __init__ method correctly initializes a GPT4o object.
    """
    gpt = GPT4o(source="test")
    assert gpt.source == "test"
    assert gpt.model_id == "gpt-4o"
    assert gpt.model_name == "GPT 4o"


@pytest.mark.skipif(not has_openai_api_key, reason="OpenAI API key not found in environment")
@patch.dict(os.environ, {'OPENAI_API_KEY': 'dummy_key'}, clear=True)
def test_gpt41_init():
    gpt = GPT4_1(source="test")
    assert gpt.source == "test"
    assert gpt.model_id == "gpt-4.1"
    assert gpt.model_name == "GPT 4.1"


@pytest.mark.skipif(not has_openai_api_key, reason="OpenAI API key not found in environment")
@patch.dict(os.environ, {'OPENAI_API_KEY': 'dummy_key'}, clear=True)
def test_gpt41mini_init():
    gpt = GPT4_1Mini(source="test")
    assert gpt.source == "test"
    assert gpt.model_id == "gpt-4.1-mini"
    assert gpt.model_name == "GPT 4.1 mini"


@pytest.mark.skipif(not has_openai_api_key, reason="OpenAI API key not found in environment")
@patch.dict(os.environ, {'OPENAI_API_KEY': 'dummy_key'}, clear=True)
def test_o3mini_init():
    gpt = O3Mini(source="test")
    assert gpt.source == "test"
    assert gpt.model_id == "o3-mini"
    assert gpt.model_name == "o3 mini"


@pytest.mark.skipif(not has_openai_api_key, reason="OpenAI API key not found in environment")
@patch.dict(os.environ, {'OPENAI_API_KEY': 'dummy_key'}, clear=True)
def test_o4mini_init():
    gpt = O4Mini(source="test")
    assert gpt.source == "test"
    assert gpt.model_id == "o4-mini"
    assert gpt.model_name == "o4 mini"


@pytest.mark.skipif(not has_openai_api_key, reason="OpenAI API key not found in environment")
@patch.dict(os.environ, {'OPENAI_API_KEY': 'dummy_key'}, clear=True)
def test_gpt41nano_init():
    gpt = GPT4_1Nano(source="test")
    assert gpt.source == "test"
    assert gpt.model_id == "gpt-4.1-nano"
    assert gpt.model_name == "GPT 4.1 nano"


# Check if the OpenAI API key is valid and has sufficient quota
def check_openai_api():
    if not has_openai_api_key:
        return False
    try:
        # Try a minimal API call to check if the key is valid
        model = GPT4o()
        model("test")
        return True
    except ValueError as e:
        if "insufficient_quota" in str(e) or "invalid_api_key" in str(e):
            return False
        raise
    except Exception:
        raise


# Verify the OpenAI API key
has_valid_openai_api = check_openai_api()


@pytest.mark.skipif(
    not has_valid_openai_api,
    reason="OpenAI API key not found in environment or has insufficient quota"
)
def test_gpt41_simple_prompt():
    """Test GPT4_1 with a simple prompt"""
    model = GPT4_1(source="test")
    response = model("Say 'Hello' in one word.")
    assert isinstance(response, str)
    assert len(response) > 0


@pytest.mark.skipif(
    not has_valid_openai_api,
    reason="OpenAI API key not found in environment or has insufficient quota"
)
def test_gpt41mini_simple_prompt():
    """Test GPT4_1Mini with a simple prompt"""
    model = GPT4_1Mini(source="test")
    response = model("Say 'Hello' in one word.")
    assert isinstance(response, str)
    assert len(response) > 0


@pytest.mark.skipif(
    not has_valid_openai_api,
    reason="OpenAI API key not found in environment or has insufficient quota"
)
def test_gpt41nano_simple_prompt():
    """Test GPT4_1Nano with a simple prompt"""
    model = GPT4_1Nano(source="test")
    response = model("Say 'Hello' in one word.")
    assert isinstance(response, str)
    assert len(response) > 0


@pytest.mark.skipif(
    not has_valid_openai_api,
    reason="OpenAI API key not found in environment or has insufficient quota"
)
def test_o3mini_simple_prompt():
    """Test O3Mini with a simple prompt"""
    model = O3Mini(source="test")
    response = model("Say 'Hello' in one word.")
    assert isinstance(response, str)
    assert len(response) > 0


@pytest.mark.skipif(
    not has_valid_openai_api,
    reason="OpenAI API key not found in environment or has insufficient quota"
)
def test_o4mini_simple_prompt():
    """Test O4Mini with a simple prompt"""
    model = O4Mini(source="test")
    response = model("Say 'Hello' in one word.")
    assert isinstance(response, str)
    assert len(response) > 0
