import time
from datetime import datetime, timezone
from uuid import uuid4

import boto3
from fence.src.llm.templates.models import Message, Messages
from fence.src.utils.base import setup_logging

logger = setup_logging(__name__)


class DynamoMemory:

    def __init__(
        self,
        table_name: str,
        org_uuid: str = None,
        session_id: str = None,
        region_name: str = "eu-central-1",
    ):

        # If session_id is not provided, `org_uuid` is required
        if not session_id and not org_uuid:
            raise ValueError("If session_id is not provided, org_uuid is required")

        # Connect to the DynamoDB service
        self.dynamodb = boto3.resource("dynamodb", region_name=region_name)
        self.table = self.dynamodb.Table(table_name)

        # Check if the table exists
        if self.table.table_status != "ACTIVE":
            raise ValueError(f"Table {table_name} is not active")

        # Set a session id if it is not provided
        if not session_id:
            self._set_session_id(org_uuid=org_uuid)
        else:
            self.session_id = session_id

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
            "PK": self.session_id,
            "SK": str(datetime.now(tz=timezone.utc)),
            "message": message.content,
            "role": message.role,
            "meta": {"state": state, "assets": assets},
        }

    def _set_session_id(self, org_uuid: str) -> None:
        """
        Set a session id for the current session, formatted as "O{org_uuid}#S{uuid4()}"
        This is used as a Primary Key in the DynamoDB table.
        """
        self.session_id = f"O{org_uuid}#S{uuid4()}"

    ##########
    # Saving #
    ##########

    def store_message(
        self,
        message: Message,
        state: dict | None = None,
        assets: list[str] | None = None,
    ):
        self.table.put_item(
            Item=self._format_message_for_dynamo_db(
                message=message, state=state, assets=assets
            )
        )

    ###########
    # Loading #
    ###########

    def get_history(self) -> (Messages, dict, list[str]):

        logger.info(f"Getting history for session {self.session_id}")
        response = self.table.query(
            KeyConditionExpression=boto3.dynamodb.conditions.Key("PK").eq(
                self.session_id
            )
        )
        items = response.get("Items", [])
        messages = Messages(
            messages=[
                Message(
                    role=item["role"],
                    content=item["message"],
                )
                for item in items
            ]
        )

        # Get last state and assets (go backwards till not None)
        state, assets = None, None
        for item in items[::-1]:
            if state is None:
                state = item["meta"]["state"]
            if assets is None:
                assets = item["meta"]["assets"]
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
        session_id="new_session_id",
    )

    # Store the user message
    memory.store_message(message=user_message)

    # Store the assistant message, along with a state
    memory.store_message(
        message=assistant_message,
        state=None,
    )

    # Test
    message_history, last_state, last_assets = memory.get_history()
    logger.info(f"Message history: {message_history}")
    logger.info(f"State: {last_state}")
    logger.info(f"Assets: {last_assets}")
