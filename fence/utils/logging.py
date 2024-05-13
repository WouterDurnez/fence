"""
Logging utils
"""

import logging
import os

DEFAULT_LOGGING_FORMAT = "[%(levelname)s][%(name)s.%(funcName)s:%(lineno)d] %(message)s"


class ColorFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[1;34m",  # Bold blue
        "INFO": "\033[1;32m",  # Bold green
        "WARNING": "\033[1;33m",  # Bold yellow
        "ERROR": "\033[1;31m",  # Bold red
        "CRITICAL": "\033[1;35m",  # Bold magenta
    }
    EMOJIS = {
        "DEBUG": "🔵",
        "INFO": "✅",
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
        formatted_message = f"[{time}] [{record.levelname}] {self.EMOJIS.get(record.levelname, '')} {formatted_message}"

        return f"{level_color}{formatted_message}{reset_color}"


def setup_logging(name: str = "root", serious_mode: bool = True):
    """
    Setup logging for use in lambdas.
    :param name: name of the logger
    :return: logger instance
    """
    # Set the default log level to INFO if not provided in the environment variable
    log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, None)

    # Validate and fallback to WARNING if an invalid log level is provided
    if not isinstance(log_level, int):
        log_level = logging.WARNING
        logging.error(
            "Invalid log level in environment variable, defaulting to WARNING"
        )

    # Set the root logger level
    logging.root.setLevel(log_level)

    # Set a logger with the provided name, and add a stream handler with the custom formatter
    logger = logging.getLogger(name)

    # Disable propagation to avoid duplicate logs
    logger.propagate = False

    # Remove all handlers associated with the logger object.
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create a stream handler
    handler = logging.StreamHandler()

    # Set the formatter
    if not serious_mode:
        handler.setFormatter(ColorFormatter(DEFAULT_LOGGING_FORMAT))
    else:
        handler.setFormatter(logging.Formatter(DEFAULT_LOGGING_FORMAT))

    # Clear any existing handlers
    logger.handlers.clear()
    logger.addHandler(handler)

    # Set the logger level
    logger.setLevel(level=log_level)

    return logger


if __name__ == "__main__":

    logger = setup_logging(__name__)
    logger.info("Test information message")
    logger.warning("Test warning message")
    logger.error("Test error message")
    logger.critical("Test critical message")
