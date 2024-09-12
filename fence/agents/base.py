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

    def __init__(self, model: LLM = None, description: str | None = None):
        """
        Initialize the Agent object.

        :param model: An LLM model object.
        :param description: A description of the agent.
        """
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
agent_name = "{self.__class__.__name__}"
agent_description = "{self.description or self.__doc__}"
"""

        return toml_string
