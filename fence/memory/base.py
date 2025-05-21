"""
Memory classes.
"""

import logging
from abc import ABC, abstractmethod

from fence.models.base import LLM
from fence.templates.messages import MessagesTemplate
from fence.templates.models import (
    Content,
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


class BaseMemory(ABC):
    """
    Base memory class for agents.
    """

    ###########
    # Setters #
    ###########

    @abstractmethod
    def add_message(self, role: str, content: str | Content, meta: dict | None = None):
        """
        Add a message to the memory buffer.
        :param role: The role of the message.
        :param content: The content of the message.
        :param meta: The meta information of the message.
        """
        pass

    def add_user_message(self, content: str | Content):
        """
        Add a user message to the memory buffer.
        :param content: The content of the message, either a string or a Content object.
        """
        self.add_message(role="user", content=content)

    def add_assistant_message(self, content: str | Content):
        """
        Add an assistant message to the memory buffer.
        :param content: The content of the message, either a string or a Content object.
        """
        self.add_message(role="assistant", content=content)

    @abstractmethod
    def set_system_message(self, content: str):
        """
        Add a system message to the memory buffer.
        :param content: The content of the system message.
        """
        pass

    @abstractmethod
    def get_messages(self):
        """
        Get the messages from the memory buffer.
        """
        pass

    @abstractmethod
    def get_system_message(self):
        """
        Get the system message from the memory buffer.
        :return: The system message as a string or None if not set.
        """
        pass

    ###########
    # Helpers #
    ###########

    def to_messages_template(self):
        """
        Convenience method to convert the memory to a MessagesTemplate object.
        """
        template = MessagesTemplate(
            source=Messages(
                messages=self.get_messages(), system=self.get_system_message()
            )
        )
        return template

    def generate_summary(self, model: LLM, n_messages: int = 10):
        """
        Generate a summary of the memory.
        :param model: The model to use for the summary.
        """

        # Get the last n messages
        messages = self.get_messages()[-n_messages:]

        # Create a prompt for the summary
        prompt = f"""
        Generate a summary of the following conversation:
        {messages}

        Do not return anything other than the summary.
        """
        response = model.invoke(prompt)
        return response

    def generate_title(self, model: LLM, n_messages: int | None = None):
        """
        Generate a title for the memory.
        :param model: The model to use for the title.
        :param n_messages: The number of messages to use for the title. If None, all messages are used. If negative, the last n messages are used. If positive, the first n messages are used.
        """
        if n_messages == 0:
            logger.warning("`n_messages` cannot be 0, using all messages")
            n_messages = None

        # Get messages based on n_messages parameter
        messages = (
            self.get_messages()[:n_messages]
            if n_messages and n_messages > 0
            else (
                self.get_messages()[-n_messages:]
                if n_messages and n_messages < 0
                else self.get_messages()
            )
        )

        # Create a prompt for the title
        prompt = f"""
        Generate a title for the following conversation:
        {messages}

        Do not return anything other than the title. Use 3 words maximum.
        """
        response = model.invoke(prompt)
        return response


class FleetingMemory(BaseMemory):
    """
    A simple memory context for the agent, only kept in memory (lol).
    """

    def __init__(self):
        """Initialize the memory object."""
        self.system = None
        self.messages = []

    def add_message(self, role: str, content: str | Content, meta: dict | None = None):
        """
        Add a message to the memory buffer.
        :param role: The role of the message.
        :param content: The content of the message, either a string or a Content object.
        :param meta: The meta information of the message, not used in FleetingMemory.
        """
        # If we got a str, it's a TextContent object
        if isinstance(content, str):
            content = TextContent(text=content)

        if role == "system":
            self.system = content
        elif role in ["user", "assistant"]:
            self.messages.append(Message(role=role, content=[content]))
        else:
            raise ValueError(
                f"Role must be 'system', 'user', or 'assistant'. Got {role}"
            )

    def set_system_message(self, content: str):
        """
        Add a system message to the memory buffer.
        :param content: The content of the system message.
        """
        self.add_message(role="system", content=content)

    def add_messages(self, messages: list[Message]):
        """
        Add a list of messages to the memory buffer.
        :param messages: The list of messages to add.
        """
        self.messages.extend(messages)

    def get_messages(self):
        """
        Get the messages from the memory buffer.
        :return: The list of messages.
        """
        return self.messages

    def get_system_message(self):
        """
        Get the system message from the memory buffer.
        :return: The system message as a string or None if not set.
        """
        if not hasattr(self, "system") or self.system is None:
            return None

        # Extract text from TextContent if needed
        if isinstance(self.system, TextContent):
            return self.system.text
        return self.system


if __name__ == "__main__":

    logger = setup_logging(log_level="debug", are_you_serious=False)

    # Create memory object
    memory = FleetingMemory()

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
    logger.info(f"System message <{type(system).__name__}>: {system}")
    logger.info(f"Messages <{type(messages).__name__}>: {messages}")
