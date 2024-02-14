import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from rich.logging import Console, RichHandler

load_dotenv()

CONF_DIR = Path(__file__).resolve().parent.parent / "conf"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

LOGGING_FORMAT = "%(message)s"
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
LOG_LEVEL = getattr(logging, LOG_LEVEL)


def setup_logging(log_level: str = LOG_LEVEL):
    # Set up rich logging
    logging.basicConfig(
        level=log_level,
        format=LOGGING_FORMAT,
        datefmt="[%X]",
        handlers=[RichHandler(console=Console(width=160))],
    )
    logger = logging.getLogger("rich")
    logger.setLevel(log_level)
    return logger


def time_it(f=None, threshold: int = 300):
    """
    Timer decorator: shows how long execution of function took.
    :param f: function to measure
    :param threshold: threshold for warning
    :return: /
    """
    # Check if the decorator is used without parentheses
    if f is None:
        return lambda func: time_it(func, threshold=threshold)

    setup_logging()
    log = logging.getLogger(__name__)

    def timed(*args, **kwargs):
        t1 = time.time()
        res = f(*args, **kwargs)
        t2 = time.time()
        duration = round(t2 - t1, 2)

        class_name = args[0].__class__.__name__  # Get the class name from the first argument
        object_name = args[0].name if hasattr(args[0], "name") else None
        method_name = f.__name__
        message = f"Method <{class_name}{(f'[{object_name}]' if object_name else '')}.{method_name}> took {duration} s to execute."
        if duration > threshold:
            log.warning(message)
        else:
            log.info(message)

        return res

    return timed
