"""
Logging utils
"""

import logging
import logging.config
from enum import Enum

DEFAULT_LOGGING_FORMAT = "[%(levelname)s][%(name)s.%(funcName)s:%(lineno)d] %(message)s"

#####################
# Base configuration #
#####################


class ColorFormatter(logging.Formatter):
    """
    A custom formatter that adds joy to the log messages.
    """

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

    def format(self, record):
        level_color = self.COLORS.get(record.levelname, "\033[0m")  # Default to reset
        reset_color = "\033[0m"

        # Format the message with the LOGGING_FORMAT
        formatted_message = super().format(record)

        # Add timestamp, level, and color to the message
        time = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        formatted_message = f"[{time}] [{record.levelname}] {self.EMOJIS.get(record.levelname, '')} [{record.module}.{record.funcName}:{record.lineno}] {formatted_message}"

        return f"{level_color}{formatted_message}{reset_color}"


class LogLevels(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


def setup_logging(
    name: str = "root", log_level: str = None, are_you_serious: bool = None
):
    """
    Setup logging for use in applications.
    :param name: name of the logger
    :param log_level: log level as a string (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    :param are_you_serious: whether to use the serious mode
    :return: logger instance
    """

    # If log level is `kill`, disable all logging
    if log_level and log_level.lower().strip() == "kill":
        logging.disable(logging.CRITICAL)
        return

    # Base configuration
    config_dict = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "simple": {
                "format": "%(asctime)s [%(levelname)s] [%(name)s.%(funcName)s:%(lineno)d] %(message)s"
            },
            "color": {"()": ColorFormatter},
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

    # Set the log level if it is not provided
    if log_level is not None:
        if log_level.upper() not in LogLevels.__members__:
            raise ValueError(
                f"Invalid log level: {log_level}. Should be one of {', '.join(LogLevels.__members__)}."
            )
        config_dict["root"]["level"] = log_level.upper()
        config_dict["handlers"]["console"]["level"] = log_level.upper()

    # Set the serious mode if it is not provided
    if are_you_serious is not None:
        config_dict["handlers"]["console"]["formatter"] = (
            "simple" if are_you_serious else "color"
        )

    # Configure the logging module with the config file
    logging.config.dictConfig(config_dict)

    # Suppress logging from boto3 and botocore
    logging.getLogger("boto3").setLevel(logging.CRITICAL)
    logging.getLogger("botocore").setLevel(logging.CRITICAL)
    logging.getLogger("urllib3").setLevel(logging.CRITICAL)

    # Get the logger
    logger = logging.getLogger(name)

    return logger


if __name__ == "__main__":

    logger = setup_logging(log_level="debug", are_you_serious=False)
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
