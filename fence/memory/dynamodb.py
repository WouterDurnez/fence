"""
Memory classes.
"""

import logging
import uuid
from datetime import datetime, timezone

import boto3

from fence.memory.base import BaseMemory
from fence.models.anthropic.claude import Claude35Sonnet
from fence.templates.models import (
    Content,
    ImageBlob,
    ImageContent,
    Message,
    Messages,
    TextContent,
    ToolResultBlock,
    ToolResultContent,
    ToolResultContentBlockText,
    ToolUseBlock,
    ToolUseContent,
)
from fence.utils.logger import setup_logging

logger = logging.getLogger(__name__)


class DynamoDBMemory(BaseMemory):
    """
    A dynamoDB-based memory context for the agent.
    """

    def __init__(
        self,
        table_name: str,
        primary_key_value: str,
        sort_key_value: str | None = None,
        primary_key_name: str = "PK",
        sort_key_name: str = "SK",
        sort_key_value_prefix: str | None = None,
        region_name: str = "eu-central-1",
        source: str = None,
        extra_fields: dict | None = None,
    ):
        """
        Initialize the DynamoDBMemory object.
        :param str table_name: The name of the DynamoDB table.
        :param str primary_key_name: The name of the partition key. Required.
        :param str primary_key_value: The partition key. Required.
        :param str sort_key_name: The name of the sort key. Optional.
        :param str sort_key_value: The sort key. Optional. Defaults to a UUID.
        :param str sort_key_value_prefix: The prefix of the sort key value. Optional.
        :param str region_name: The AWS region name.
        :param str source: A custom identifier for what created the memory object.
        :param dict extra_fields: Extra fields to store in the DynamoDB table with each entry.
        """

        self.table_name = table_name
        self.source = source

        # Connect to the DynamoDB service
        try:
            self.db = boto3.resource("dynamodb", region_name=region_name)
            self.table = self.db.Table(self.table_name)
        except Exception as e:
            logger.error(
                f"Failed to connect to DynamoDB table {table_name} - {e}", exc_info=True
            )
            raise ValueError(f"Failed to connect to DynamoDB table {table_name}")

        # Set PK and SK
        self.primary_key_name = primary_key_name
        self.sort_key_name = sort_key_name
        self.sort_key_value_prefix = sort_key_value_prefix

        # Secondary key holds the session ID
        self.primary_key_value = primary_key_value
        self.sort_key_value = sort_key_value if sort_key_value else str(uuid.uuid4())
        self.sort_key_formatted = f"{self.sort_key_value_prefix}#{self.sort_key_value}"

        # Extra fields
        self.extra_fields = extra_fields if extra_fields else {}

        # Check if the table exists
        if self.table.table_status != "ACTIVE":
            raise ValueError(f"Table {table_name} is not active")

    def add_message(self, role: str, content: str | Content, meta: dict | None = None):
        """
        Add a message to the memory buffer.
        :param role: The role of the message.
        :param content: The content of the message.
        :param meta: The meta information of the message.
        """

        # If we got a str, it's a TextContent object
        if isinstance(content, str):
            content = TextContent(text=content)

        # Store the message in DynamoDB
        dynamo_response = self.table.put_item(
            Item=self._format_message_for_dynamo_db(
                role=role, content=content, meta=meta
            )
        )
        status_code = dynamo_response["ResponseMetadata"]["HTTPStatusCode"]
        if status_code != 200:
            logger.error(f"Failed to store message in DynamoDB: {dynamo_response}")
            raise ValueError(f"Failed to store message in DynamoDB: {dynamo_response}")

    def add_user_message(self, content: str | Content):
        """
        Add a user message to the memory buffer.
        :param content: The content of the message, either a string or a Content object.
        """
        self.add_message(role="user", content=content)

    def add_assistant_message(self, content: str):
        self.add_message(role="assistant", content=content)

    def set_system_message(self, content: str):
        self.add_message(role="system", content=content)

    def add_messages(self, messages: list[Message]):
        """
        Add a list of messages to the memory buffer.
        :param messages: The list of messages.
        """
        for message in messages:
            self.add_message(role=message.role, content=message.content)

    def _get_messages(self):
        """
        Get ALL messages from the dynamoDB.
        """

        logger.debug(f"Getting history for session {self.primary_key_name}")
        response = self.table.query(
            KeyConditionExpression=boto3.dynamodb.conditions.Key(
                f"{self.primary_key_name}"
            ).eq(self.primary_key_value)
            & boto3.dynamodb.conditions.Key(f"{self.sort_key_name}").begins_with(
                self.sort_key_formatted
            )
        )
        items = response.get("Items", [])

        # Parse items into content objects based on type in the Message object
        for item in items:
            type = item["Message"]["type"]
            match type:
                case "text":
                    item["Message"] = TextContent(text=item["Message"]["text"])
                case "image":
                    item["Message"] = ImageContent(image=item["Message"]["image"])
                case "image_blob":
                    item["Message"] = ImageBlob(
                        image_blob=item["Message"]["image_blob"]
                    )
                case "toolResult":
                    item["Message"] = ToolResultContent(
                        content=item["Message"]["content"]
                    )
                case "toolUse":
                    item["Message"] = ToolUseContent(content=item["Message"]["content"])
                case _:
                    raise ValueError(f"Unknown message type: {type}")

        # Sort all messages by the timestamp part of the sort key (after the #)
        items.sort(key=lambda x: x[self.sort_key_name].split("#")[-1])

        # Create a Messages object
        messages = Messages(
            messages=[
                Message(
                    role=item["Role"],
                    content=[item["Message"]],
                )
                for item in items
                if item["Role"] in ["user", "assistant"]
            ]
        )
        # Get the latest system message by sorting on the timestamp in the sort key
        system_messages = [
            (item["Message"], item[self.sort_key_name])
            for item in items
            if item["Role"] == "system"
        ]
        if system_messages:
            # Sort by the timestamp part of the sort key (after the #)
            latest_system = max(system_messages, key=lambda x: x[1].split("#")[-1])[0]
            messages.system = (
                latest_system.text
                if isinstance(latest_system, TextContent)
                else latest_system
            )
        else:
            messages.system = None

        # Log the messages
        logger.debug(
            f"<SYSTEM>:\t{messages.system}\n"
            + "\n".join(
                [
                    f"<{message.role.upper()}>:\t{message.content}"
                    for message in messages.messages
                ]
            )
        )

        return messages

    def get_messages(self):
        """
        Get the messages from the memory buffer.
        """
        messages = self._get_messages()
        return messages.messages

    def get_system_message(self):
        """
        Get the system message from the memory buffer.
        """
        messages = self._get_messages()
        return messages.system

    ###########
    # Helpers #
    ###########

    def _format_message_for_dynamo_db(
        self,
        role: str,
        content: Content | str,
        meta: dict | None = None,
    ) -> dict:
        """
        Format a message for storage in DynamoDB.
        :param role: The role of the message.
        :param content: The content of the message.
        :param meta: The meta information of the message.
        :return: The formatted message.
        """
        if isinstance(content, str):
            content = TextContent(text=content)

        item = {
            f"{self.primary_key_name}": self.primary_key_value,
            f"{self.sort_key_name}": self.sort_key_formatted
            + f"#{datetime.now(timezone.utc).isoformat()}",
            "Message": content.model_dump(),
            "Role": role,
            "Meta": meta,
            "Source": self.source,
        }

        # Add extra fields
        if self.extra_fields:
            item |= self.extra_fields

        return item


if __name__ == "__main__":

    setup_logging(log_level="debug")

    # Create a DynamoDBMemory object
    memory = DynamoDBMemory(
        table_name="agent_test",
        primary_key_name="PK",
        primary_key_value="org_pid_1234567890#user_id_1234567890",
        # primary_key_value="test_a",
        extra_fields={"test_a": "test_b"},
    )

    # Add a system message
    memory.set_system_message("This is a system message")

    # Add a later system message
    memory.set_system_message("This is a later system message")

    # Add a user message
    memory.add_user_message("This is a user message")

    # Add an assistant message
    memory.add_assistant_message("This is an assistant message")

    # Add a later user message
    memory.add_user_message("This is a later user message")

    # Add a later assistant message
    memory.add_assistant_message("This is a later assistant message")

    # Add a later system message
    memory.set_system_message("This is a much later system message")

    # Add a tool use block
    memory.add_user_message(
        ToolUseContent(
            content=ToolUseBlock(
                toolUseId="1234567890", input={"test": "test"}, name="test"
            )
        )
    )
    # Add a tool result message
    memory.add_assistant_message(
        ToolResultContent(
            content=ToolResultBlock(
                content=[
                    ToolResultContentBlockText(text="This is a tool result message")
                ],
                toolUseId="1234567890",
                status="success",
            )
        )
    )

    # Print the messages
    messages = memory.get_messages()
    system = memory.get_system_message()
    logger.info(f"System message <{type(system)}>: {system}")
    logger.info(f"Messages <{type(messages)}>: {messages}")

    # Generate a summary
    model = Claude35Sonnet(model="claude-3-5-sonnet-20240620")
    summary = memory.generate_summary(model=model)
    logger.info(f"\n\nSummary: {summary}\n\n")

    # Generate a title
    title = memory.generate_title(model=model)
    logger.info(f"\n\nTitle: {title}\n\n")
