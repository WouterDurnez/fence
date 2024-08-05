"""
Base class for LLMs
"""

from abc import ABC, abstractmethod
from typing import Any, Callable

from fence.templates.messages import Messages

# Type alias for the logging callback function
# The logging callback function should accept two arguments:
# - a dictionary of metrics and their values
# - a list of tags
LogCallback = Callable[[dict, list], None]

# Global variable to hold the logging callback function
_global_log_callback: LogCallback | None = None

# Default tags for logging
_global_log_tags: dict | None = None


def register_log_callback(log_callback: Callable):
    """Register a global logging callback function."""
    global _global_log_callback
    _global_log_callback = log_callback


def register_log_tags(tags: dict):
    """Register global tags for logging."""
    global _global_log_tags
    _global_log_tags = tags


def get_log_callback():
    """Retrieve the global logging callback function."""
    return _global_log_callback


def get_log_tags():
    """Retrieve the global logging tags."""
    return _global_log_tags


class LLM(ABC):
    """
    Base class for LLMs
    """

    model_name = None
    llm_name = None
    inference_type = None
    source = None

    def __call__(self, prompt: str | Any, **kwargs) -> str:
        return self.invoke(prompt, **kwargs)

    @abstractmethod
    def invoke(self, prompt: str | Messages, **kwargs) -> str:
        raise NotImplementedError
