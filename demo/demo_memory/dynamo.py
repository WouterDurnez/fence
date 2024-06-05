from datetime import datetime, timezone
from uuid import uuid4

import boto3
from fence.templates.messages import Message, Messages
from fence.utils.logger import setup_logging

logger = setup_logging(__name__)


class DynamoMemory:

    def __init__(
        self,
        table_name: str,
        org_uuid: str = None,
        session_id: str = None,
        region_name: str = "eu-central-1",
    ):
        """
        Initialize the DynamoMemory class.
        :param table_name: The name of the DynamoDB table.
        :param org_uuid: The UUID of the organization.
        :param session_id: The ID of the session. Note that the PK in the DynamoDB table will be formatted as
            "O{org_uuid}#S{session_id}".
        :param region_name: The name of the region.
        """

        # Connect to the DynamoDB service
        self.dynamodb = boto3.resource("dynamodb", region_name=region_name)
        self.table = self.dynamodb.Table(table_name)

        # Check if the table exists
        if self.table.table_status != "ACTIVE":
            raise ValueError(f"Table {table_name} is not active")

        # Set a session id if it is not provided
        self.session_id = str(uuid4()) if session_id is None else session_id

        # Set the primary key
        self.primary_key = f"O{org_uuid}#S{self.session_id}"

    def _format_message_for_dynamo_db(
        self,
        message: Message,
        state: dict | None = None,
        assets: list[str] | None = None,
    ) -> dict:
        """
        Format a message for storage in DynamoDB.
        :param message: The message to store.
        :param: state: A state dictionary to keep track of variables.
        :param: assets: A list of assets.
        """
        return {
            "SessionId": self.primary_key,
            "Timestamp": str(datetime.now(tz=timezone.utc)),
            "Message": message.content,
            "Role": message.role,
            "Meta": {"state": state, "assets": assets},
        }

    #################
    # Apply history #
    #################

    def apply_history(self, message: Message):
        """
        Apply history to the current session. Gets the history of the sessions, appends the message to the messages,
        and returns the updated messages, state, and assets.
        :param message: The last message in the history.
        """
        messages, state, assets = self.get_history()
        messages.messages.append(message)
        return messages, state, assets

    ##########
    # Saving #
    ##########

    def store_message(
        self,
        message: Message,
        state: dict | None = None,
        assets: list[str] | None = None,
    ):

        dynamo_response = self.table.put_item(
            Item=self._format_message_for_dynamo_db(
                message=message, state=state, assets=assets
            )
        )
        status_code = dynamo_response["ResponseMetadata"]["HTTPStatusCode"]
        if status_code != 200:
            logger.error(f"Failed to store message in DynamoDB: {dynamo_response}")
            raise ValueError(f"Failed to store message in DynamoDB: {dynamo_response}")

    ###########
    # Loading #
    ###########

    def get_history(self) -> tuple[Messages, dict, list[str]]:
        """
        Get the history of the current session.
        """

        logger.info(f"Getting history for session {self.primary_key}")
        response = self.table.query(
            KeyConditionExpression=boto3.dynamodb.conditions.Key("SessionId").eq(
                self.primary_key
            )
        )
        items = response.get("Items", [])
        messages = Messages(
            messages=[
                Message(
                    role=item["Role"],
                    content=item["Message"],
                )
                for item in items
            ]
        )

        # Log the messages
        logger.info(
            "Messages:\n"
            + "\n".join(
                [
                    f"<{message.role.upper()}>:\t{message.content}"
                    for message in messages.messages
                ]
            )
        )

        # Get last state and assets (go backwards till not None)
        state, assets = None, None
        for item in items[::-1]:
            if state is None:
                state = item["Meta"]["state"]
            if assets is None:
                assets = item["Meta"]["assets"]
            if state is not None and assets is not None:
                break
        return messages, state, assets


if __name__ == "__main__":

    user_message = Message(
        role="user",
        content="Hello, how are you?",
    )

    assistant_message = Message(
        role="assistant",
        content="I'm doing well, thank you. How can I help you today?",
    )

    # Create a DynamoMemory object
    memory = DynamoMemory(
        table_name="chat_memory",
        org_uuid="12345678",
        session_id="1",
    )

    # Store the user message
    memory.store_message(message=user_message)

    # Store the assistant message, along with a state
    memory.store_message(
        message=assistant_message,
        state=None,
    )

    # Store a message with a state and assets
    memory.store_message(
        message=Message(
            role="assistant",
            content="I can help you with that.",
        ),
        state={"intent": "help"},
        assets=["asset1", "asset2"],
    )

    # Test
    message_history, last_state, last_assets = memory.get_history()
    logger.info(f"Message history: {message_history}")
    logger.info(f"State: {last_state}")
    logger.info(f"Assets: {last_assets}")
