"""
Bedrock Claude model tests (pre gen 3)
"""

import json
from unittest.mock import MagicMock

import pytest

from fence.models.base import register_log_callback
from fence.models.bedrock.claude import ClaudeBase, ClaudeInstant, ClaudeV2


@pytest.fixture
def mock_boto_client(mocker):
    """
    Fixture to mock the boto3 client.
    """
    mock_client = mocker.patch("boto3.client")
    return mock_client


def test_claude_base_initialization(mock_boto_client):
    """
    Test the initialization of ClaudeBase class.
    """
    model = ClaudeBase(source="test_source")
    assert model.source == "test_source"
    assert model.metric_prefix == ""
    assert model.logging_tags == {}
    assert model.model_kwargs == {"temperature": 0.01, "max_tokens_to_sample": 2048}
    assert model.region == "eu-central-1"
    assert model.client == mock_boto_client.return_value


def test_claude_instant_initialization(mock_boto_client):
    """
    Test the initialization of ClaudeInstant class.
    """
    model = ClaudeInstant(source="test_source")
    assert model.source == "test_source"
    assert model.metric_prefix == ""
    assert model.logging_tags == {}
    assert model.model_kwargs == {"temperature": 0.01, "max_tokens_to_sample": 2048}
    assert model.region == "eu-central-1"
    assert model.client == mock_boto_client.return_value
    assert model.model_id == "anthropic.claude-instant-v1"
    assert model.model_name == "ClaudeInstant"


def test_claude_v2_initialization(mock_boto_client):
    """
    Test the initialization of ClaudeV2 class.
    """
    model = ClaudeV2(source="test_source")
    assert model.source == "test_source"
    assert model.metric_prefix == ""
    assert model.logging_tags == {}
    assert model.model_kwargs == {"temperature": 0.01, "max_tokens_to_sample": 2048}
    assert model.region == "eu-central-1"
    assert model.client == mock_boto_client.return_value
    assert model.model_id == "anthropic.anthropic.claude-v2"
    assert model.model_name == "ClaudeV2"


def test_invoke_method(mock_boto_client, mocker):
    """
    Test the invoke method of ClaudeInstant class.
    """
    model = ClaudeInstant(source="test_source")

    # Mock the response from the AWS service
    mock_response = {
        "ResponseMetadata": {
            "HTTPHeaders": {
                "x-amzn-bedrock-input-token-count": "10",
                "x-amzn-bedrock-output-token-count": "20",
            }
        },
        "body": MagicMock(),
    }
    mock_response["body"].read.return_value = json.dumps(
        {"completion": "test completion"}
    ).encode()
    mock_boto_client.return_value.invoke_model.return_value = mock_response

    # Patch the log callback
    mocker.patch("fence.models.base.get_log_callback", return_value=MagicMock())

    result = model.invoke(prompt="test prompt")

    assert result == "test completion"
    mock_boto_client.return_value.invoke_model.assert_called_once()


def test_invoke_method_with_logging(mock_boto_client, mocker):
    """
    Test the invoke method with logging callback.
    """
    model = ClaudeInstant(source="test_source")

    # Mock the response from the AWS service
    mock_response = {
        "ResponseMetadata": {
            "HTTPHeaders": {
                "x-amzn-bedrock-input-token-count": "10",
                "x-amzn-bedrock-output-token-count": "20",
            }
        },
        "body": MagicMock(),
    }
    mock_response["body"].read.return_value = json.dumps(
        {"completion": "test completion"}
    ).encode()
    mock_boto_client.return_value.invoke_model.return_value = mock_response

    # Register and patch the log callback
    log_callback = MagicMock()
    register_log_callback(log_callback)
    mocker.patch("fence.models.base.get_log_callback", return_value=log_callback)

    result = model.invoke(prompt="test prompt")

    assert result == "test completion"
    mock_boto_client.return_value.invoke_model.assert_called_once()
    log_callback.assert_called_once()


def test_invoke_method_exception(mock_boto_client):
    """
    Test the invoke method to handle exceptions correctly.
    """
    model = ClaudeInstant(source="test_source")

    # Mock an exception being raised by the AWS service
    mock_boto_client.return_value.invoke_model.side_effect = Exception("Test exception")

    with pytest.raises(
        ValueError, match="Error raised by bedrock service: Test exception"
    ):
        model.invoke(prompt="test prompt")
