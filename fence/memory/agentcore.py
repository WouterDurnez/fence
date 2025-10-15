"""
AgentCore memory class.
"""

import logging

from fence.memory.base import BaseMemory
from fence.templates.models import Content, Message, Messages, TextContent

logger = logging.getLogger(__name__)


class AgentCoreMemory(BaseMemory):
    """
    Memory implementation for Bedrock Agent Core.
    Automatically flushes TextContent messages to the memory service.
    """

    def __init__(self, agentcore_memory, client, actor_id: str, session_id: str):
        self.agentcore_memory = agentcore_memory
        self.client = client
        self.actor_id = actor_id
        self.session_id = session_id
        self.system = None
        self.messages = []

    def add_message(self, role: str, content: str | Content, meta: dict | None = None):
        if isinstance(content, str):
            content = TextContent(text=content)

        if role == "system":
            self.system = content
            return

        if not isinstance(content, TextContent):
            raise TypeError(
                f"Only TextContent is supported in AgentCoreMemory for now. Got {type(content).__name__}"
            )

        if role not in ["user", "assistant"]:
            raise ValueError(f"Unsupported role: {role}")

        # Auto-flush this message to AgentCore
        self.client.create_event(
            memory_id=self.agentcore_memory["id"],
            actor_id=self.actor_id,
            session_id=self.session_id,
            messages=[(content.text, role.upper())],
        )

    def set_system_message(self, content: str):
        self.add_message("system", content)

    def get_messages(self):
        print(f"requested all messages: {len(self.messages)}")

        messages = self._get_messages()
        return messages.messages

    def get_system_message(self):
        if isinstance(self.system, TextContent):
            return self.system.text
        return self.system

    def _get_messages(self):
        """
        Get ALL messages from the memory client.
        """

        if not self.agentcore_memory or not self.actor_id or not self.session_id:
            raise ValueError(
                "Memory, actor_id, and session_id must be set before fetching messages."
            )

        events = self.client.list_events(
            memory_id=self.agentcore_memory["id"],
            actor_id=self.actor_id,
            session_id=self.session_id,
        )

        events.sort(key=lambda x: x["eventTimestamp"])

        # Create a Messages object
        messages = Messages(
            messages=[
                Message(
                    role=event["payload"][0]["conversational"]["role"].lower(),
                    content=[
                        TextContent(
                            text=event["payload"][0]["conversational"]["content"][
                                "text"
                            ]
                        )
                    ],
                )
                for event in events
            ],
            system=None,
        )

        logger.info(
            f"<SYSTEM>:\t{messages.system}\n"
            + "\n".join(
                [
                    f"<{message.role.upper()}>:\t{message.content}"
                    for message in messages.messages
                ]
            )
        )

        logger.info(f"messages type: {type(messages)}")

        return messages

    def _fetch_recent_messages(self, max_results: int = 20):
        return self.client.list_events(
            memory_id=self.agentcore_memory["id"],
            actor_id=self.actor_id,
            session_id=self.session_id,
            max_results=max_results,
        )
