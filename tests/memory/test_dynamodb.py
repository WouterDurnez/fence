from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from fence.memory.dynamodb import DynamoDBMemory
from fence.templates.messages import Message

############
# Fixtures #
############


@pytest.fixture
def mock_table():
    table = MagicMock()
    table.table_status = "ACTIVE"
    return table


@pytest.fixture
def mock_boto3():
    with patch("fence.memory.dynamodb.boto3") as mock_boto3:
        # Set up the mock table
        mock_table = MagicMock()
        mock_table.table_status = "ACTIVE"

        # Set up the mock resource
        mock_resource = MagicMock()
        mock_resource.Table.return_value = mock_table
        mock_boto3.resource.return_value = mock_resource

        yield mock_boto3


@pytest.fixture
def memory(mock_boto3):
    return DynamoDBMemory(
        table_name="test_table",
        primary_key_value="test-session",
        sort_key_value_prefix="agent1",
        region_name="us-east-1",
    )


#########
# Tests #
#########


def test_init_success(mock_boto3):
    memory = DynamoDBMemory(
        table_name="test_table",
        primary_key_value="test-session",
        sort_key_value_prefix="agent1",
    )
    assert memory.table_name == "test_table"
    assert memory.primary_key_value == "test-session"
    assert memory.sort_key_value_prefix == "agent1"
    mock_boto3.resource.assert_called_once_with("dynamodb", region_name="eu-central-1")


def test_init_failure_inactive_table():
    with patch("fence.memory.dynamodb.boto3") as mock_boto3:
        mock_table = MagicMock()
        mock_table.table_status = "CREATING"
        mock_resource = MagicMock()
        mock_resource.Table.return_value = mock_table
        mock_boto3.resource.return_value = mock_resource

        with pytest.raises(ValueError, match="Table test_table is not active"):
            DynamoDBMemory(table_name="test_table", primary_key_value="test-session")


def test_init_failure_connection_error():
    with patch("fence.memory.dynamodb.boto3") as mock_boto3:
        mock_boto3.resource.side_effect = Exception("Connection failed")

        with pytest.raises(ValueError, match="Failed to connect to DynamoDB table"):
            DynamoDBMemory(table_name="test_table", primary_key_value="test-session")


def test_add_message_success(memory, mock_boto3):
    table = mock_boto3.resource.return_value.Table.return_value
    table.put_item.return_value = {"ResponseMetadata": {"HTTPStatusCode": 200}}

    memory.add_message(role="user", content="test message")

    table.put_item.assert_called_once()
    call_args = table.put_item.call_args[1]["Item"]
    assert call_args["Role"] == "user"
    assert call_args["Message"]["type"] == "text"
    assert call_args["Message"]["text"] == "test message"
    assert call_args["PK"] == "test-session"


def test_add_message_failure(memory, mock_boto3):
    table = mock_boto3.resource.return_value.Table.return_value
    table.put_item.return_value = {"ResponseMetadata": {"HTTPStatusCode": 400}}

    with pytest.raises(ValueError, match="Failed to store message in DynamoDB"):
        memory.add_message(role="user", content="test message")


def test_add_user_message(memory):
    with patch.object(memory, "add_message") as mock_add_message:
        memory.add_user_message("test message")
        mock_add_message.assert_called_once_with(role="user", content="test message")


def test_add_assistant_message(memory):
    with patch.object(memory, "add_message") as mock_add_message:
        memory.add_assistant_message("test message")
        mock_add_message.assert_called_once_with(
            role="assistant", content="test message"
        )


def test_set_system_message(memory):
    with patch.object(memory, "add_message") as mock_add_message:
        memory.set_system_message("test message")
        mock_add_message.assert_called_once_with(role="system", content="test message")


def test_get_messages(memory, mock_boto3):
    mock_messages = [
        {
            "Role": "system",
            "Message": {"type": "text", "text": "system message"},
            "SK": "agent1#test-session#2024-01-01T00:00:00+00:00#2024-01-01T00:00:00+00:00",
        },
        {
            "Role": "user",
            "Message": {"type": "text", "text": "user message"},
            "SK": "agent1#test-session#2024-01-01T00:00:01+00:00#2024-01-01T00:00:01+00:00",
        },
        {
            "Role": "assistant",
            "Message": {"type": "text", "text": "assistant message"},
            "SK": "agent1#test-session#2024-01-01T00:00:02+00:00#2024-01-01T00:00:02+00:00",
        },
    ]

    table = mock_boto3.resource.return_value.Table.return_value
    table.query.return_value = {"Items": mock_messages}

    messages = memory.get_messages()

    table.query.assert_called_once()
    assert table.query.call_count == 1
    # Check that the sort key prefix is used in the query
    sort_key_prefix = f"{memory.sort_key_value_prefix}#{memory.sort_key_value}"
    assert sort_key_prefix in memory.sort_key_formatted

    assert len(messages) == 2  # system message should not be included
    assert messages[0].role == "user"
    assert messages[0].content[0].text == "user message"
    assert messages[1].role == "assistant"
    assert messages[1].content[0].text == "assistant message"


def test_get_system_message(memory, mock_boto3):
    mock_messages = [
        {
            "Role": "system",
            "Message": {"type": "text", "text": "system message"},
            "SK": "agent1#test-session#2024-01-01T00:00:00+00:00#2024-01-01T00:00:00+00:00",
        },
        {
            "Role": "system",
            "Message": {"type": "text", "text": "newer system message"},
            "SK": "agent1#test-session#2024-01-01T00:00:01+00:00#2024-01-01T00:00:01+00:00",
        },
        {
            "Role": "user",
            "Message": {"type": "text", "text": "user message"},
            "SK": "agent1#test-session#2024-01-01T00:00:02+00:00#2024-01-01T00:00:02+00:00",
        },
    ]

    table = mock_boto3.resource.return_value.Table.return_value
    table.query.return_value = {"Items": mock_messages}

    system_message = memory.get_system_message()

    table.query.assert_called_once()
    assert table.query.call_count == 1
    # Check that the sort key prefix is used in the query
    sort_key_prefix = f"{memory.sort_key_value_prefix}#{memory.sort_key_value}"
    assert sort_key_prefix in memory.sort_key_formatted

    assert (
        system_message == "newer system message"
    )  # Should get the latest system message


@pytest.fixture
def mock_datetime():
    with patch("fence.memory.dynamodb.datetime") as mock_datetime:
        mock_now = datetime(2024, 1, 1, tzinfo=timezone.utc)
        mock_datetime.now.return_value = mock_now
        yield mock_datetime


def test_format_message_for_dynamo_db(memory, mock_datetime):
    formatted_message = memory._format_message_for_dynamo_db(
        role="user", content="test message", meta={"key": "value"}
    )

    sort_key_base = memory.sort_key_formatted

    assert formatted_message == {
        "PK": "test-session",
        "SK": f"{sort_key_base}#2024-01-01T00:00:00+00:00",
        "Message": {"type": "text", "text": "test message"},
        "Role": "user",
        "Meta": {"key": "value"},
        "Source": None,
    }


def test_add_messages(memory):
    with patch.object(memory, "add_message") as mock_add_message:
        messages = [
            Message(role="user", content="user message"),
            Message(role="assistant", content="assistant message"),
        ]

        memory.add_messages(messages)

        assert mock_add_message.call_count == 2
        mock_add_message.assert_any_call(role="user", content="user message")
        mock_add_message.assert_any_call(role="assistant", content="assistant message")
