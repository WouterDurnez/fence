"""
Tests for the GeminiBase class and related functionality.
"""

import os
from unittest.mock import Mock, patch, MagicMock

import pytest

from fence.models.gemini.gemini import Gemini, GeminiBase, MessagesMixin
from fence.templates.messages import Message, Messages

# Check if Google Gemini API key is present
has_gemini_api_key = os.environ.get("GOOGLE_GEMINI_API_KEY") is not None


class MockGeminiModel(GeminiBase, MessagesMixin):
    """
    Mock implementation of the GeminiBase class for testing purposes.
    """

    def __init__(self, **kwargs):
        super().__init__(source="test", api_key="dummy_key", **kwargs)
        self.model_id = "mock-gemini-model"

    def _invoke(self, prompt: str | Messages, **kwargs) -> dict:
        """
        Simulate model invocation.
        """
        return {
            "candidates": [{"content": {"parts": [{"text": "mocked response"}]}}],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 20},
        }


def test_gemini_base_invoke_with_empty_prompt():
    """
    Test case for the invoke method of the GeminiBase class with an empty prompt.
    """
    gemini = MockGeminiModel()
    with pytest.raises(ValueError):
        gemini.invoke(prompt="")


def test_gemini_base_invoke_with_none_prompt():
    """
    Test case for the invoke method of the GeminiBase class with a None prompt.
    """
    gemini = MockGeminiModel()
    with pytest.raises(ValueError):
        gemini.invoke(prompt=None)


def test_gemini_base_invoke_with_empty_messages_prompt():
    """
    Test case for the invoke method of the GeminiBase class with an empty Messages prompt.
    """
    gemini = MockGeminiModel()
    messages = Messages(system="Respond with a sarcastic tone", messages=[])
    with pytest.raises(ValueError):
        gemini.invoke(prompt=messages)


def test_gemini_base_invoke_with_string_prompt():
    """
    Test case for the invoke method of the GeminiBase class with a string prompt.
    """
    gemini = MockGeminiModel()
    response = gemini.invoke(prompt="Hello, how are you today?")
    assert response == "mocked response"


def test_gemini_base_invoke_with_messages_prompt():
    """
    Test case for the invoke method of the GeminiBase class with a Messages prompt.
    """
    gemini = MockGeminiModel()
    messages = Messages(
        system="Respond with a sarcastic tone",
        messages=[Message(role="user", content="Hello, how are you today?")],
    )
    response = gemini.invoke(prompt=messages)
    assert response == "mocked response"


def test_gemini_base_invoke_with_invalid_prompt():
    """
    Test case for the invoke method of the GeminiBase class with an invalid prompt.
    """
    gemini = MockGeminiModel()
    with pytest.raises(ValueError):
        gemini.invoke(prompt=123)


@patch("fence.models.gemini.gemini.requests.post")
def test_gemini_api_key_sent_in_header_not_params(mock_post):
    """
    Test that the Gemini API key is sent via the x-goog-api-key header,
    not as a URL query parameter, to prevent key leakage in logs.
    """
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": "mocked response"}]}}],
        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 20},
    }
    mock_post.return_value = mock_response

    gemini = Gemini(model_id="gemini-2.0-flash", source="test", api_key="test-secret-key")
    gemini.invoke(prompt="Hello")

    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args

    # API key must be in headers
    assert call_kwargs.kwargs["headers"]["x-goog-api-key"] == "test-secret-key"

    # API key must NOT be in URL params
    assert "params" not in call_kwargs.kwargs or call_kwargs.kwargs.get("params") is None


@patch("fence.models.gemini.gemini.requests.post")
def test_gemini_request_has_timeout(mock_post):
    """
    Test that Gemini API requests include an explicit timeout
    to prevent indefinite hangs.
    """
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": "mocked response"}]}}],
        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 20},
    }
    mock_post.return_value = mock_response

    gemini = Gemini(model_id="gemini-2.0-flash", source="test", api_key="test-key")
    gemini.invoke(prompt="Hello")

    call_kwargs = mock_post.call_args
    assert "timeout" in call_kwargs.kwargs
    assert call_kwargs.kwargs["timeout"] > 0
