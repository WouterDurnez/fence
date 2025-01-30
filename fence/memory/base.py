"""
Memory classes.
"""

import logging
from abc import ABC, abstractmethod

from fence.templates.messages import MessagesTemplate
from fence.templates.models import Message, Messages

logger = logging.getLogger(__name__)


class BaseMemory(ABC):
    """
    Base memory class for agents.
    """

    ###########
    # Setters #
    ###########

    @abstractmethod
    def add_message(self, role: str, content: str):
        """
        Add a message to the memory buffer.
        """
        pass

    def add_user_message(self, content: str):
        """
        Add a user message to the memory buffer.
        """
        self.add_message(role="user", content=content)

    def add_assistant_message(self, content: str):
        """
        Add an assistant message to the memory buffer.
        """
        self.add_message(role="assistant", content=content)

    @abstractmethod
    def set_system_message(self, content: str):
        """
        Add a system message to the memory buffer.
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


class FleetingMemory(BaseMemory):
    """
    A simple memory context for the agent, only kept in memory (lol).
    """

    def __init__(self):
        """Initialize the memory object."""
        self.system = None
        self.messages = []

    def add_message(self, role: str, content: str):

        if role == "system":
            self.system = content
        elif role in ["user", "assistant"]:
            self.messages.append(Message(role=role, content=content))
        else:
            raise ValueError(
                f"Role must be 'system', 'user', or 'assistant'. Got {role}"
            )

    def set_system_message(self, content: str):
        self.add_message(role="system", content=content)

    def add_messages(self, messages: list[Message]):
        self.messages.extend(messages)

    def get_messages(self):
        return self.messages

    def get_system_message(self):
        return self.system if hasattr(self, "system") else None


if __name__ == "__main__":

    # Create memory object
    memory = FleetingMemory()

    # Add a system message
    memory.set_system_message("This is a system message")

    # Add a user message
    memory.add_user_message("This is a user message")

    # Add an assistant message
    memory.add_assistant_message("This is an assistant message")

    # Print the messages
    logger.info(
        f"System message <{type(memory.get_system_message())}>: {memory.get_system_message()}"
    )
    logger.info(f"Messages <{type(memory.get_messages())}>: {memory.get_messages()}")
