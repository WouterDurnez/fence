"""
Memory classes.
"""

import logging
import uuid
from datetime import datetime, timezone

import boto3

from fence.memory.base import BaseMemory
from fence.templates.models import Message, Messages
from fence.utils.logger import setup_logging

logger = logging.getLogger(__name__)


class DynamoDBMemory(BaseMemory):
    """
    A dynamoDB-based memory context for the agent.
    """

    def __init__(
        self,
        table_name: str,
        primary_key_name: str = "PK",
        primary_key_value: str | None = None,
        primary_key_value_prefix: str = "",
        sort_key_name: str = "SK",
        region_name: str = "eu-central-1",
        source: str = None,
    ):
        """
        Initialize the DynamoDBMemory object.
        :param str table_name: The name of the DynamoDB table.
        :param str primary_key_name: The name of the partition key.
        :param str primary_key_value: The partition key.
        :param str primary_key_value_prefix: The prefix for the partition key. Useful for tenant isolation.
        :param str sort_key_name: The name of the sort key.
        :param str region_name: The AWS region name.
        :param str source: A custom identifier for what created the memory object.
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

        # Primary key counts as session ID
        self.primary_key_value = (
            primary_key_value if primary_key_value else str(uuid.uuid4())
        )
        self.primary_key_formatted = (
            f"{primary_key_value_prefix}{self.primary_key_value}"
        )

        # Check if the table exists
        if self.table.table_status != "ACTIVE":
            raise ValueError(f"Table {table_name} is not active")

    def add_message(self, role: str, content: str, meta: dict | None = None):
        """
        Add a message to the memory buffer.
        :param role: The role of the message.
        :param content: The content of the message.
        :param meta: The meta information of the message.
        """

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

    def add_user_message(self, content: str):
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
            ).eq(self.primary_key_formatted)
        )
        items = response.get("Items", [])
        messages = Messages(
            messages=[
                Message(
                    role=item["Role"],
                    content=item["Message"],
                )
                for item in items
                if item["Role"] in ["user", "assistant"]
            ]
        )
        messages.system = next(
            (item["Message"] for item in items if item["Role"] == "system"),
            None,
        )

        # Log the messages
        logger.debug(
            "Messages:\n"
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
        content: str,
        meta: dict | None = None,
    ) -> dict:
        """
        Format a message for storage in DynamoDB.
        :param role: The role of the message.
        :param content: The content of the message.
        :param meta: The meta information of the message.
        :return: The formatted message.
        """
        return {
            f"{self.primary_key_name}": self.primary_key_formatted,
            f"{self.sort_key_name}": datetime.now(timezone.utc).isoformat(),
            "Message": content,
            "Role": role,
            "Meta": meta,
            "Source": self.source,
        }


if __name__ == "__main__":

    setup_logging(log_level="info")

    # Create a DynamoDBMemory object
    memory = DynamoDBMemory(
        table_name="fence_test",
        primary_key_name="session",
        primary_key_value="08f1201a-d9cd-4ef7-9b63-c26d2c824576",
        # primary_key_value="test_a",
    )

    # # Add a system message
    # memory.set_system_message("This is a system message")
    #
    # # Add a user message
    # memory.add_user_message("This is a user message")
    #
    # # Add an assistant message
    # memory.add_assistant_message("This is an assistant message")

    # Print the messages
    messages = memory.get_messages()
    system = memory.get_system_message()
    logger.info(f"System message <{type(system)}>: {system}")
    logger.info(f"Messages <{type(messages)}>: {messages}")
