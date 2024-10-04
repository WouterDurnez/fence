"""
Logging utils
"""

import logging
import logging.config
from enum import Enum

from wcwidth import wcswidth

DEFAULT_LOGGING_FORMAT = "[%(levelname)s][%(name)s.%(funcName)s:%(lineno)d] %(message)s"


#####################
# Base configuration #
#####################


class LogLevels(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


#####################
# Formatter classes #
#####################


class BaseFixedWidthFormatter(logging.Formatter):
    def __init__(
        self,
        fmt: str = None,
        datefmt: str = None,
        width: int = 60,
        timestamp: bool = True,
        function_trace: bool = True,
    ):
        """
        Initialize the BaseFixedWidthFormatter object.

        :param fmt: The format string for the formatter.
        :param datefmt: The date format string for the formatter.
        :param width: The width of the log message.
        :param timestamp: Whether to include the timestamp in the log message.
        :param function_trace: Whether to include the function line in the log message.
        """
        super().__init__(fmt, datefmt)
        self.width = width
        self.timestamp = timestamp
        self.func_name = function_trace

    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record.
        """
        # Save the original format
        format_orig = self._style._fmt

        # Replace the original format with one customized by logging level
        self._style._fmt = "%(message)s"

        # Call the original formatter class
        formatted_message = super().format(record)

        # Restore the original format
        self._style._fmt = format_orig

        # Get the non-message part
        non_message_part = self.get_non_message_part(record)

        # Calculate padding
        visual_width = self.get_visual_width(non_message_part)
        padding_needed = max(0, self.width - visual_width)

        # Format the final message with proper indentation
        final_message = self.format_final_message(
            non_message_part, formatted_message, padding_needed
        )

        return self.apply_styling(final_message, record)

    def get_non_message_part(self, record: logging.LogRecord) -> str:
        """
        Get the non-message part of the log record, such as the timestamp, log level, and function trace.
        """
        parts = []
        if self.timestamp:
            parts.append(self.formatTime(record, "%Y-%m-%d %H:%M:%S"))
        parts.append(record.levelname)
        if self.func_name:
            parts.append(f"{record.module}.{record.funcName}:{record.lineno}")
        return " ".join(f"[{part}]" for part in parts if part)

    def get_visual_width(self, text: str):
        """
        Get the visual width of the text.
        """
        return wcswidth(text)

    def format_final_message(
        self, non_message_part: str, formatted_message: str, padding_needed: int
    ) -> str:
        """
        Format the final message with proper indentation for multiline messages.
        """
        lines = formatted_message.splitlines()

        # First line combines the non-message part with the first part of the message
        first_line = f"{non_message_part} {' ' * padding_needed}{lines[0]}"

        # Calculate indentation for subsequent lines (aligning them with the start of the message)
        # The indentation should match the width of the non-message part plus the padding
        message_start_column = self.get_visual_width(non_message_part) + padding_needed
        message_indent = " " * message_start_column

        # Handle subsequent lines (if the message spans multiple lines)
        if len(lines) > 1:
            overflow_lines = "\n".join(f"{message_indent}{line}" for line in lines[1:])
            final_message = f"{first_line}\n{overflow_lines}"
        else:
            final_message = first_line

        return final_message

    def apply_styling(self, message: str, record: logging.LogRecord) -> str:
        """
        Apply styling to the message. This method should be overridden by subclasses.
        """
        return message  # No styling in base class


class ColorFormatter(BaseFixedWidthFormatter):
    COLORS = {
        "DEBUG": "\033[1;34m",  # Bold blue
        "INFO": "\033[1;32m",  # Bold green
        "WARNING": "\033[1;33m",  # Bold yellow
        "ERROR": "\033[1;31m",  # Bold red
        "CRITICAL": "\033[1;35m",  # Bold magenta
    }
    EMOJIS = {
        "DEBUG": "🔵",
        "INFO": "ℹ️",
        "WARNING": "😬",
        "ERROR": "❌",
        "CRITICAL": "🚨",
    }

    def __init__(self, color_dict: dict = None, emoji_dict: dict = None, **kwargs):
        """
        Initialize the ColorFormatter object.

        :param color_dict: A dictionary of log level to color mappings.
        :param emoji_dict: A dictionary of log level to emoji mappings.
        """
        super().__init__(**kwargs)
        self.COLORS = color_dict or self.COLORS
        self.EMOJIS = emoji_dict or self.EMOJIS

    def get_non_message_part(self, record: logging.LogRecord) -> str:
        """
        Get the non-message part of the log record, such as the timestamp, log level, and function trace. Here, we add emojis because RE
        """
        parts = []
        if self.timestamp:
            parts.append(self.formatTime(record, "%Y-%m-%d %H:%M:%S"))
        parts.append(f"{self.EMOJIS.get(record.levelname, '')} {record.levelname}")
        if self.func_name:
            parts.append(f"{record.module}.{record.funcName}:{record.lineno}")
        return " ".join(f"[{part}]" for part in parts if part)

    def apply_styling(self, message, record):
        """
        Apply styling to the message. Here, we add color to the log level.
        """
        level_color = self.COLORS.get(record.levelname, "\033[0m")  # Default to reset
        reset_color = "\033[0m"
        return f"{level_color}{message}{reset_color}"


def setup_logging(
    name: str = "root",
    log_level: str = None,
    width: int = 60,
    timestamp: bool = True,
    function_trace: bool = True,
    serious_mode: bool = True,
):
    # Base configuration
    config_dict = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "simple": {
                "()": BaseFixedWidthFormatter,
                "fmt": "%(asctime)s [%(levelname)s] [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
                "width": width,
                "timestamp": timestamp,
                "function_trace": function_trace,
            },
            "color": {
                "()": ColorFormatter,
                "fmt": "%(asctime)s [%(levelname)s] [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
                "width": width,
                "timestamp": timestamp,
                "function_trace": function_trace,
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "simple",
                "stream": "ext://sys.stdout",
            }
        },
        "root": {"level": "INFO", "handlers": ["console"]},
    }

    # Set the log level if it is provided
    if log_level is not None:
        if log_level.upper() not in LogLevels.__members__:
            raise ValueError(
                f"Invalid log level: {log_level}. Should be one of {', '.join(LogLevels.__members__)}."
            )
        config_dict["root"]["level"] = log_level.upper()
        config_dict["handlers"]["console"]["level"] = log_level.upper()

    # Set the formatter based on serious_mode
    config_dict["handlers"]["console"]["formatter"] = (
        "simple" if serious_mode else "color"
    )

    # Configure the logging module with the config dictionary
    logging.config.dictConfig(config_dict)

    return logging.getLogger(name)


if __name__ == "__main__":
    logger = setup_logging(log_level="debug", serious_mode=True, timestamp=True)
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")

    logger.info(
        "This is a very long info message. It spans many lines. It is designed to overflow."
        * 10
    )

    logger = setup_logging(
        log_level="debug", serious_mode=False, timestamp=True, function_trace=True
    )
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")

    logger.info(
        "This is a very long info message.\n It spans many lines.\n It is designed to overflow."
        * 10
    )
