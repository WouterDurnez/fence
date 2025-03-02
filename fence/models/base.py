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
LogCallback = Callable[[dict, dict], None]

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


#######################
# Base class for LLMs #
#######################


class LLM(ABC):
    """
    Base class for LLMs
    """

    model_id: str
    model_name: str
    inference_type: str | None = None
    source: str | None = None
    logging_tags: dict = {}

    def __init__(
        self,
        source: str | None = None,
        metric_prefix: str | None = None,
        extra_tags: dict | None = None,
        **kwargs,
    ):
        """
        Initialize an LLM

        :param str source: An indicator of where (e.g., which feature) the model is operating from
        :param str|None metric_prefix: Prefix for the metric names
        :param dict|None extra_tags: Additional tags to add to the logging tags
        :param **kwargs: Additional keyword arguments
        """

        # LLM parameters
        self.source = source

        # Logging parameters
        self.metric_prefix = metric_prefix
        self.logging_tags = get_log_tags() or {}
        self.logging_tags.update(extra_tags or {})

    def __call__(self, prompt: str | Any, **kwargs) -> str:
        return self.invoke(prompt, **kwargs)

    @abstractmethod
    def invoke(self, prompt: str | Messages, **kwargs) -> str:
        raise NotImplementedError

    @staticmethod
    def _check_if_prompt_is_valid(prompt: str | Messages):
        """
        Check if the prompt is a valid user message or Messages object
        """

        # Prompt should be a string or Messages object
        if not isinstance(prompt, (str, Messages)):
            raise ValueError("Prompt must be a string or a Messages object!")

        # Prompt should not be empty
        if isinstance(prompt, str) and not prompt.strip():
            raise ValueError("Prompt cannot be empty string!")
        if isinstance(prompt, Messages) and not prompt.messages:
            raise ValueError("Prompt needs at least one user message!")

    def _log_callback(self, **metrics):
        """
        Log the callback arguments
        :param metrics: metrics to log
        """
        # Log all metrics if a log callback is registered
        if log_callback := get_log_callback():

            # Use precomputed log_prefix for efficiency
            log_metrics = {
                f"{self.metric_prefix + '.' if self.metric_prefix else ''}{metric_key}": metric_value
                for metric_key, metric_value in metrics.items()
            }
            log_metrics[
                f"{self.metric_prefix + '.' if self.metric_prefix else ''}invocation"
            ] = 1
            log_args = [
                # Add metrics
                log_metrics,
                # Add tags
                self.logging_tags,
            ]
            # Log metrics
            log_callback(*log_args)


class InvokeMixin:
    """
    Mixin for LLMs that can be invoked
    """

    @abstractmethod
    def invoke(self, prompt: str | Messages, **kwargs) -> str:
        """
        Invoke the LLM
        :param prompt: User message or Messages object
        :param kwargs: Additional keyword arguments
        :return: Response message
        """
        raise NotImplementedError


class StreamMixin:
    """
    Mixin for LLMs that can stream responses
    """

    @abstractmethod
    def stream(self, prompt: str | Messages, **kwargs) -> str:
        """
        Stream the LLM responses
        :param prompt: User message or Messages object
        :param kwargs: Additional keyword arguments
        :return: Stream of responses
        """
        raise NotImplementedError


##########
# Mixins #
##########


class MessagesMixin:
    """
    Mixin for LLMs that take a Messages object as input

    This mixin provides a method to count the number of words in a list of messages
    """

    @staticmethod
    def count_words_in_messages(messages: Messages):
        """
        Count the number of words in a list of messages. Takes all roles into account. Type must be 'text'
        :param messages: list of messages
        :return: word count (int)
        """

        # Get user and assistant messages
        user_assistant_messages = messages.messages

        # Get system message
        system_messages = messages.system

        # Initialize word count
        word_count = 0

        # Loop through messages
        for message in user_assistant_messages:

            # Get content
            content = message.content

            # Content is either a string, or a list of content objects
            if isinstance(content, str):
                word_count += len(content.split())
            elif isinstance(content, list):
                for content_object in content:
                    if content_object.type == "text":
                        word_count += len(content_object.text.split())

        # Add system message to word count
        if system_messages:
            word_count += len(system_messages.split())

        return word_count
