"""
Memory classes.
"""

from abc import ABC, abstractmethod

from fence import Message, Messages


class BaseMemory(Messages, ABC):
    """
    Base memory class for agents.
    """

    @abstractmethod
    def add_message(self, role: str, content: str):
        """
        Add a message to the memory buffer.
        """
        pass

    @abstractmethod
    def add_user_message(self, content: str):
        """
        Add a user message to the memory buffer.
        """
        pass

    @abstractmethod
    def add_assistant_message(self, content: str):
        """
        Add an assistant message to the memory buffer.
        """
        pass

    @abstractmethod
    def set_system_message(self, content: str):
        """
        Add a system message to the memory buffer.
        """
        pass


class FleetingMemory(BaseMemory):
    """
    A simple memory context for the agent, only kept in memory (lol).
    """

    # Override the messages attribute, avoiding pydantic validation
    messages: list[Message] = []

    def add_message(self, role: str, content: str):

        if role == "system":
            self.system = content
        elif role in ["user", "assistant"]:
            self.messages.append(Message(role=role, content=content))
        else:
            raise ValueError(
                f"Role must be 'system', 'user', or 'assistant'. Got {role}"
            )

    def add_user_message(self, content: str):
        self.add_message(role="user", content=content)

    def add_assistant_message(self, content: str):
        self.add_message(role="assistant", content=content)

    def set_system_message(self, content: str):
        self.add_message(role="system", content=content)

    def add_messages(self, messages: list[Message]):
        self.messages.extend(messages)

    def get_messages(self):
        return self.messages

    def get_system_message(self):
        return self.system
