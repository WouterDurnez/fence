"""
Base Agent class
"""

from abc import abstractmethod

from fence import LLM, setup_logging
from fence.links import logger as link_logger

logger = setup_logging(__name__, log_level="info", serious_mode=False)

# Suppress the link logger
link_logger.setLevel("INFO")


class BaseAgent:
    """Base Agent class"""

    def __init__(
        self,
        identifier: str | None = None,
        model: LLM = None,
        description: str | None = None,
        environment: dict | None = None,
    ):
        """
        Initialize the Agent object.

        :param identifier: An identifier for the agent. If none is provided, the class name will be used.
        :param model: An LLM model object.
        :param description: A description of the agent.
        :param environment: A dictionary of environment variables to pass to delegates and tools.
        """
        self.environment = environment or {}
        self.identifier = identifier or self.__class__.__name__
        self.model = model
        self.description = description

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
        toml_string = f"""[[agents]]
agent_name = "{self.identifier}"
agent_description = "{self.description or self.__doc__}"
"""

        return toml_string
