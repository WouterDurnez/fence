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

    def __init__(self, model: LLM = None):
        """
        Initialize the Agent object.

        :param model: An LLM model object.
        """
        self.model = model

    @abstractmethod
    def run(self, prompt: str) -> str:
        """
        Run the agent

        :param prompt: The initial prompt to feed to the LLM
        """
        raise NotImplementedError
