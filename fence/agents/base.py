"""
Base Agent class
"""

import logging
from abc import abstractmethod
from enum import Enum

from fence.links import logger as link_logger
from fence.memory.base import BaseMemory, FleetingMemory
from fence.models.base import LLM

logger = logging.getLogger(__name__)

# Suppress the link logger
link_logger.setLevel("INFO")


# Parts of an agentic response
class AgentLogType(Enum):
    THOUGHT = "thought"
    ACTION = "action"
    DELEGATE = "delegate"
    ANSWER = "answer"
    OBSERVATION = "observation"


class BaseAgent:
    """Base Agent class"""

    def __init__(
        self,
        identifier: str | None = None,
        model: LLM = None,
        description: str | None = None,
        memory: BaseMemory | None = None,
        environment: dict | None = None,
        log_agentic_response: bool = True,
        are_you_serious: bool = False,
    ):
        """
        Initialize the Agent object.

        :param identifier: An identifier for the agent. If none is provided, the class name will be used.
        :param model: An LLM model object.
        :param description: A description of the agent.
        :param environment: A dictionary of environment variables to pass to delegates and tools.
        :param memory: A memory object to store messages and system messages.
        :param log_agentic_response: A flag to determine if the agent's responses should be logged.
        :param are_you_serious: A flag to determine if the log message should be printed in a frivolous manner.
        """
        self.environment = environment or {}
        self.identifier = identifier or self.__class__.__name__
        self.model = model
        self.description = description

        # Set the log configuration
        self.log_agentic_response = log_agentic_response
        self.are_you_serious = are_you_serious

        # Memory setup
        self.memory = memory or FleetingMemory()

        # Set system message
        self._system_message = None

    @abstractmethod
    def run(self, prompt: str) -> str:
        """
        Run the agent

        :param prompt: The initial prompt to feed to the LLM
        """
        raise NotImplementedError

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
        """Clear or reset the agent's memory context."""

        # Check if there are any messages in the memory
        self.memory.messages = self.memory.get_messages()

        # Check if there is a system message in the memory
        self.memory.system = self.memory.get_system_message()

        # If no system message is present, add a new one
        if not self.memory.system:
            self.memory.set_system_message(content=self._system_message)

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
            AgentLogType.DELEGATE: "\033[93m",  # Yellow
            AgentLogType.ANSWER: "\033[91m",  # Red
            AgentLogType.OBSERVATION: "\033[95m",  # Purple
            "reset": "\033[0m",  # Reset
        }
        emojis = {
            AgentLogType.THOUGHT: "💭",
            AgentLogType.ACTION: "🛠️",
            AgentLogType.DELEGATE: "🤝",
            AgentLogType.ANSWER: "🎯",
            AgentLogType.OBSERVATION: "🔍",
        }

        tag = f"[{type.value}]"
        if self.are_you_serious:
            print(f"{tag} {message}")
        else:
            color = colors.get(type, colors["reset"])
            emoji = emojis.get(type, "")
            reset = colors["reset"]
            print(f"{emoji} {color}{tag}{reset} {message}")
