"""
Logging utils
"""

import logging

DEFAULT_LOGGING_FORMAT = "[%(levelname)s][%(name)s.%(funcName)s:%(lineno)d] %(message)s"


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
        formatted_message = f"[{time}] [{record.levelname}] {self.EMOJIS.get(record.levelname, '')} [{record.module}.{record.funcName}:{record.lineno}] {formatted_message}"

        return f"{level_color}{formatted_message}{reset_color}"
#
#
#
# def setup_logging(name: str = "root", log_level: str = None, serious_mode: bool = None):
#     """
#     Setup logging for use in applications.
#     :param name: name of the logger
#     :param log_level: log level as a string (DEBUG, INFO, WARNING, ERROR, CRITICAL)
#     :param serious_mode: whether to use the serious mode
#     :return: logger instance
#     """
#
#     # Set the log level if it is not provided
#     if log_level is not None:
#         fence.config.LOG_LEVEL = getattr(logging, log_level, logging.WARNING)
#
#     # Set the serious mode if it is not provided
#     if serious_mode is not None:
#         fence.config.SERIOUS_MODE = serious_mode
#
#     logger = logging.getLogger(name)
#     logger.setLevel(fence.config.LOG_LEVEL)
#     logger.propagate = True
#
#     handler = logging.StreamHandler()
#     if fence.config.SERIOUS_MODE:
#         formatter = logging.Formatter(DEFAULT_LOGGING_FORMAT)
#     else:
#         formatter = ColorFormatter()
#     handler.setFormatter(formatter)
#     logger.addHandler(handler)
#
#     return logger


if __name__ == "__main__":

    fence.config.SERIOUS_MODE = False
    logger = logging.getLogger(__name__)
    logger.info("Test information message")
    logger.warning("Test warning message")
    logger.error("Test error message")
    logger.critical("Test critical message")
