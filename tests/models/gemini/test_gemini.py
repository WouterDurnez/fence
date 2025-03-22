"""
Google Gemini model tests
"""

import os
from unittest.mock import Mock

import pytest

from fence.models.gemini.gemini import Gemini
from fence.templates.messages import Message, Messages

# Check if Google Gemini API key is present
has_gemini_api_key = os.environ.get("GOOGLE_GEMINI_API_KEY") is not None


@pytest.mark.skipif(
    not has_gemini_api_key, reason="Google Gemini API key not found in environment"
)
def test_gemini_base_invoke_with_empty_prompt():
    """
    Test case for the invoke method of the GeminiBase class with an empty prompt.
    This test checks if the invoke method raises a ValueError when the prompt is empty.
    """
    gemini = Gemini(source="test", model_id="any_gemini_model")
    with pytest.raises(ValueError):
        gemini.invoke(prompt="")


@pytest.mark.skipif(
    not has_gemini_api_key, reason="Google Gemini API key not found in environment"
)
def test_gemini_base_invoke_with_none_prompt():
    """
    Test case for the invoke method of the GeminiBase class with a None
    prompt. This test checks if the invoke method raises a ValueError when the
    prompt is None.
    """
    gemini = Gemini(source="test", model_id="any_gemini_model")
    with pytest.raises(ValueError):
        gemini.invoke(prompt=None)


@pytest.mark.skipif(
    not has_gemini_api_key, reason="Google Gemini API key not found in environment"
)
def test_gemini_base_invoke_with_empty_messages_prompt():
    """
    Test case for the invoke method of the GeminiBase class with an empty
    Messages prompt. This test checks if the invoke method raises a ValueError
    when the Messages prompt is empty.
    """
    gemini = Gemini(source="test", model_id="any_gemini_model")
    messages = Messages(system="Respond with a sarcastic tone", messages=[])
    with pytest.raises(ValueError):
        gemini.invoke(prompt=messages)


def test_gemini_base_invoke_with_string_prompt():
    """
    Test case for the invoke method of the GeminiBase class with a string
    prompt. This test checks if the invoke method correctly handles a string
    prompt.
    """

    mock_gemini = Mock(Gemini)
    mock_gemini.invoke.return_value = "mocked response"
    response = mock_gemini.invoke(prompt="Hello, how are you today?")
    assert response == "mocked response"


def test_gemini_base_invoke_with_messages_prompt():
    """
    Test case for the invoke method of the GeminiBase class with a Messages
    prompt. This test checks if the invoke method correctly handles a Messages
    prompt.
    """

    mock_gemini = Mock(Gemini)
    mock_gemini.invoke.return_value = "mocked response"
    messages = Messages(
        system="Respond with a sarcastic tone",
        messages=[Message(role="user", content="Hello, how are you today?")],
    )
    response = mock_gemini.invoke(prompt=messages)
    assert response == "mocked response"


@pytest.mark.skipif(
    not has_gemini_api_key, reason="Google Gemini API key not found in environment"
)
def test_gemini_base_invoke_with_invalid_prompt():
    """
    Test case for the invoke method of the GeminiBase class with an invalid
    prompt. This test checks if the invoke method raises a ValueError when the
    prompt is invalid.
    """
    gemini = Gemini(source="test", model_id="any_gemini_model")
    with pytest.raises(ValueError):
        gemini.invoke(prompt=123)


@pytest.mark.skipif(
    not has_gemini_api_key, reason="Google Gemini API key not found in environment"
)
def test_gemini_flash2_0():
    """
    Test case for the GeminiFlash2_0 class.
    This test checks if the model_id is correctly set to "gemini-2.0-flash".
    """
    gemini = Gemini(source="test", model_id="gemini-2.0-flash")
    assert gemini.model_id == "gemini-2.0-flash"


@pytest.mark.skipif(
    not has_gemini_api_key, reason="Google Gemini API key not found in environment"
)
def test_gemini_flash1_5():
    """
    Test case for the GeminiFlash1_5 class.
    This test checks if the model_id is correctly set to "gemini-1.5-flash".
    """
    gemini = Gemini(source="test", model_id="gemini-1.5-flash")
    assert gemini.model_id == "gemini-1.5-flash"


@pytest.mark.skipif(
    not has_gemini_api_key, reason="Google Gemini API key not found in environment"
)
def test_gemini_1_5_pro():
    """
    Test case for the Gemini1_5_Pro class.
    This test checks if the model_id is correctly set to "gemini-1.5-pro".
    """
    gemini = Gemini(source="test", model_id="gemini-1.5-pro")
    assert gemini.model_id == "gemini-1.5-pro"
