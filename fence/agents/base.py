"""
Base Agent class
"""

import logging
from abc import ABC, abstractmethod
from enum import Enum

from fence.links import logger as link_logger
from fence.memory.base import BaseMemory, FleetingMemory
from fence.models.bedrock.base import BedrockBase
from fence.tools.base import BaseTool

logger = logging.getLogger(__name__)

# Suppress the link logger
link_logger.setLevel("INFO")


# Parts of an agentic response
class AgentLogType(Enum):
    START = "start"
    STOP = "stop"
    THOUGHT = "thought"
    ACTION = "action"
    DELEGATION = "delegation"
    ANSWER = "answer"
    OBSERVATION = "observation"
    TOOL_USE = "tool_use"


####################
# Base Agent class #
####################


class BaseAgent(ABC):
    """Base Agent class"""

    def __init__(
        self,
        identifier: str | None = None,
        model: BedrockBase = None,
        description: str | None = None,
        role: str | None = None,
        memory: BaseMemory | None = None,
        environment: dict | None = None,
        prefill: str | None = None,
        log_agentic_response: bool = True,
        system_message: str | None = None,
        are_you_serious: bool = False,
        tools: list[BaseTool] | None = None,
    ):
        """
        Initialize the Agent object.

        :param identifier: An identifier for the agent. If none is provided, the class name will be used.
        :param model: An LLM model object.
        :param description: A description of the agent. Used in the agent's representation.
        :param role: The role of the agent. Used to describe the agent's purpose in its own system message.
        :param environment: A dictionary of environment variables to pass to delegates and tools.
        :param memory: A memory object to store messages and system messages.
        :param prefill: A string to prefill the memory with, i.e. an assistant message.
        :param log_agentic_response: A flag to determine if the agent's responses should be logged.
        :param are_you_serious: A flag to determine if the log message should be printed in a frivolous manner.
        """
        self.environment = environment or {}
        self.identifier = identifier or self.__class__.__name__
        self.model = model
        self.description = description
        self.role = role
        self.prefill = prefill

        # Set the log configuration
        self.log_agentic_response = log_agentic_response
        self.are_you_serious = are_you_serious

        # Memory setup
        self.memory = memory or FleetingMemory()

        # Set system message
        self._system_message = system_message

        # Set tools
        self.tools = tools or []

    @abstractmethod
    def run(self, prompt: str) -> str:
        """
        Run the agent

        :param prompt: The initial prompt to feed to the LLM
        """
        raise NotImplementedError

    def get_description(self):
        """
        Returns a string description of the agent.
        """
        return self.description or self.__doc__

    def get_representation(self):
        """
        Returns a string representation of the agent.
        """
        representation = (
            f"# {self.identifier}\n\n## Description:\n{self.get_description()}\n\n"
        )

        if self.tools:
            representation += "## Tools:\n"
            for tool in self.tools:
                representation += f"{tool.get_representation()}\n"

        return representation

    def format_toml(self):
        """
        Returns a TOML-formatted key-value pair of the agent name,
        the description (docstring) of the agent.
        """

        # Preformat the arguments
        toml_string = f'''[[agents]]
agent_name = """{self.identifier}"""
agent_description = """{self.description or self.__doc__}"""
'''

        return toml_string

    def _flush_memory(self):
        """
        Clear or reset the agent's memory context.
        """

        # Check if there is a system message in the memory
        existing_system = self.memory.get_system_message()

        # Only set system message if none exists
        if not existing_system and self._system_message:
            self.memory.set_system_message(content=self._system_message)

        # If we have a prefill, add it
        if self.prefill:
            self.memory.add_message(role="assistant", content=self.prefill)

    def log(
        self,
        message: str,
        type: str | AgentLogType,
    ):
        """
        Log a part of the agent's thought process, action, delegate, or answer.

        :param message: The message to log.
        :param type: The type of log message.
        """

        # If we are not logging, return
        if not self.log_agentic_response:
            return

        # If the type is a string, convert it to an AgentLogType
        if isinstance(type, str):
            type = AgentLogType(type)

        colors = {
            AgentLogType.THOUGHT: "\033[94m",  # Blue
            AgentLogType.ACTION: "\033[92m",  # Green
            AgentLogType.DELEGATION: "\033[93m",  # Yellow
            AgentLogType.ANSWER: "\033[91m",  # Red
            AgentLogType.OBSERVATION: "\033[95m",  # Purple
            AgentLogType.TOOL_USE: "\033[38;5;208m",  # Orange
            AgentLogType.START: "\033[96m",  # Cyan
            AgentLogType.STOP: "\033[96m",  # Cyan
            "identifier": "\033[1m",  # Bold
            "reset": "\033[0m",  # Reset
        }
        emojis = {
            AgentLogType.THOUGHT: "💭",
            AgentLogType.ACTION: "🛠️",
            AgentLogType.DELEGATION: "🤝",
            AgentLogType.ANSWER: "🎯",
            AgentLogType.OBSERVATION: "🔍",
            AgentLogType.TOOL_USE: "🔧",
            AgentLogType.START: "🚀",
            AgentLogType.STOP: "🏁",
        }

        tag = f"[{type.value}]"
        if self.are_you_serious:
            print(f"{self.identifier}: {tag} {message}")
        else:
            color = colors.get(type, colors["reset"])
            emoji = emojis.get(type, "")
            reset = colors["reset"]
            identifier_color = colors["identifier"]
            print(
                f"{identifier_color}{self.identifier}{reset} {emoji} {color}{tag}{reset} {message}"
            )
