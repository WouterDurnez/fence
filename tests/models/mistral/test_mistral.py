import json
import os
from unittest.mock import Mock, patch

import pytest
import requests

from fence.models.mistral.mistral import Mistral
from fence.templates.messages import Message, Messages, MessagesTemplate

# Check if Mistral API key is present
has_mistral_api_key = os.environ.get("MISTRAL_API_KEY") is not None


@pytest.fixture
def mock_api_response():
    return {
        "id": "test_id",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "mistral-large-latest",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Test response"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


@pytest.fixture
def mock_response(mock_api_response):
    response = Mock()
    response.status_code = 200
    response.json.return_value = mock_api_response
    return response


@pytest.fixture
def mistral_model():
    with patch.dict(os.environ, {"MISTRAL_API_KEY": "test_key"}):
        return Mistral(model_id="mistral-large-latest")


@pytest.mark.skipif(
    not has_mistral_api_key, reason="Mistral API key not found in environment"
)
def test_mistral_initialization():
    """Test Mistral model initialization"""
    # Test with API key in environment
    with patch.dict(os.environ, {"MISTRAL_API_KEY": "test_key"}):
        model = Mistral(model_id="mistral-large-latest")
        assert model.model_id == "mistral-large-latest"
        assert model.model_name == "Mistral-large-latest"
        assert model.api_key == "test_key"
        assert model.url == "https://api.mistral.ai"
        assert model.chat_endpoint == "https://api.mistral.ai/v1/chat/completions"

    # Test with API key as parameter
    model = Mistral(model_id="mistral-large-latest", api_key="direct_key")
    assert model.api_key == "direct_key"

    # Test missing API key
    with pytest.raises(ValueError):
        with patch.dict(os.environ, {}, clear=True):
            Mistral(model_id="mistral-large-latest")


def test_mistral_model_parameters(mistral_model):
    """Test model parameters initialization"""
    assert mistral_model.model_kwargs["top_p"] == 1
    assert mistral_model.model_kwargs["temperature"] is None
    assert mistral_model.model_kwargs["max_tokens"] is None
    assert mistral_model.model_kwargs["stop"] is None

    # Test with custom parameters
    model = Mistral(
        model_id="mistral-large-latest",
        api_key="test_key",
        temperature=0.7,
        max_tokens=100,
        top_p=0.9,
        stop=["END"],
    )
    assert model.model_kwargs["temperature"] == 0.7
    assert model.model_kwargs["max_tokens"] == 100
    assert model.model_kwargs["top_p"] == 0.9
    assert model.model_kwargs["stop"] == ["END"]


@patch("requests.post")
def test_invoke_with_string_prompt(
    mock_post, mistral_model, mock_response, mock_api_response
):
    """Test model invocation with string prompt"""
    mock_post.return_value = mock_response

    response = mistral_model.invoke("Test prompt")

    assert response == "Test response"
    mock_post.assert_called_once()

    # Verify request format
    call_args = mock_post.call_args
    headers = call_args.kwargs["headers"]
    data = json.loads(call_args.kwargs["data"])

    assert headers["Authorization"] == "Bearer test_key"
    assert headers["Content-Type"] == "application/json"
    assert data["messages"][0]["role"] == "user"
    assert data["messages"][0]["content"] == "Test prompt"


@patch("requests.post")
def test_invoke_with_messages(mock_post, mistral_model, mock_response):
    """Test model invocation with Messages object"""
    mock_post.return_value = mock_response

    messages = Messages(
        system="System prompt",
        messages=[
            Message(role="user", content="User message"),
            Message(role="assistant", content="Assistant message"),
        ],
    )

    response = mistral_model.invoke(messages)

    assert response == "Test response"
    mock_post.assert_called_once()

    # Verify request format
    call_args = mock_post.call_args
    data = json.loads(call_args.kwargs["data"])
    assert len(data["messages"]) == 3
    assert data["messages"][0]["role"] == "system"
    assert data["messages"][0]["content"] == "System prompt"


@patch("requests.post")
def test_invoke_with_template(mock_post, mistral_model, mock_response):
    """Test model invocation with MessagesTemplate"""
    mock_post.return_value = mock_response

    messages = Messages(
        system="Respond in a {tone} tone",
        messages=[Message(role="user", content="Why is the sky {color}?")],
    )
    template = MessagesTemplate(source=messages)

    response = mistral_model.invoke(template.render(tone="sarcastic", color="blue"))

    assert response == "Test response"
    mock_post.assert_called_once()

    # Verify request format
    call_args = mock_post.call_args
    data = json.loads(call_args.kwargs["data"])
    assert data["messages"][0]["content"] == "Respond in a sarcastic tone"
    assert data["messages"][1]["content"] == "Why is the sky blue?"


@patch("requests.post")
def test_error_handling(mock_post, mistral_model):
    """Test error handling"""
    # Test API error response
    error_response = Mock()
    error_response.status_code = 400
    error_response.text = "Bad Request"
    mock_post.return_value = error_response

    with pytest.raises(ValueError) as exc_info:
        mistral_model.invoke("Test prompt")
    assert "Error raised by Mistral service" in str(exc_info.value)

    # Test network error
    mock_post.side_effect = requests.exceptions.RequestException("Network error")
    with pytest.raises(ValueError) as exc_info:
        mistral_model.invoke("Test prompt")
    assert "Error raised by Mistral service" in str(exc_info.value)


@pytest.mark.skipif(
    not has_mistral_api_key, reason="Mistral API key not found in environment"
)
def test_invalid_prompt_types(mistral_model):
    """Test handling of invalid prompt types"""
    with pytest.raises(ValueError):
        mistral_model.invoke(None)

    with pytest.raises(ValueError):
        mistral_model.invoke("")

    with pytest.raises(ValueError):
        mistral_model.invoke(123)


@patch("requests.post")
def test_metrics_logging(mock_post, mistral_model, mock_response):
    """Test metrics logging functionality"""
    mock_post.return_value = mock_response

    # Add a mock logging callback
    mock_callback = Mock()
    mistral_model._log_callback = mock_callback

    mistral_model.invoke("Test prompt")

    # Verify metrics logging
    mock_callback.assert_called_once()
    metrics = mock_callback.call_args[1]
    assert metrics["input_token_count"] == 10
    assert metrics["output_token_count"] == 5
    assert metrics["input_word_count"] == 2  # "Test prompt"
    assert metrics["output_word_count"] == 2  # "Test response"
