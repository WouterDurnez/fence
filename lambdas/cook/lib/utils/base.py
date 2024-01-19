import json
import logging
import os
import time
import tomllib
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

CONF_DIR = Path(__file__).resolve().parent.parent / "conf"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LOG_LEVEL = os.environ.get("LOG_LEVEL", "WARNING")


def setup_logging(
    log_level: str = LOG_LEVEL,
    format: str = "%(asctime)s [%(levelname)s] - %(message)s",
):
    # Convert log level string to uppercase
    log_level_str = os.environ.get("LOG_LEVEL", log_level).upper()

    # Set up logging with the specified log level
    log_level = getattr(logging, log_level_str, None)
    if not isinstance(log_level, int):
        log_level_str = "WARNING"
        log_level = logging.WARNING
        logging.error(
            "Invalid log level in environment variable, defaulting to WARNING"
        )

    # Set up logging with timestamped format
    logging.basicConfig(level=log_level, format=format)
    logging.root.setLevel(level=log_level)


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

        message = f"Function <{f.__name__}> took {duration} s to execute."
        if duration > threshold:
            log.warning(message)
        else:
            log.info(message)

        return res

    return timed


VALID_FLAVORS = ["sarcastic", "formal", "informal"]
VALID_VERBOSITY = ["shorter", "longer"]


def validate_recipe(recipe: dict):
    """
    Validate whether a recipe JSON object is valid.
    :param dict recipe: JSON object containing the recipe
    :return: True if the recipe is valid, raise ValueError otherwise
    """

    # Check if the recipe contains any invalid keys
    valid_keys = ["flavor", "verbosity", "spelling", "policy"]
    invalid_keys = [key for key in recipe.keys() if key not in valid_keys]
    if invalid_keys:
        raise ValueError(f"Invalid recipe keys: {invalid_keys}")

    # Check if the recipe contains any invalid values
    if "flavor" in recipe:
        if recipe["flavor"] not in VALID_FLAVORS:
            raise ValueError(f"Invalid flavor: {recipe['flavor']}")
    if "verbosity" in recipe:
        if recipe["verbosity"] not in VALID_VERBOSITY:
            raise ValueError(f"Invalid verbosity: {recipe['verbosity']}")
    if "spelling" in recipe:
        if not isinstance(recipe["spelling"], bool):
            raise ValueError(f"Invalid spelling: {recipe['spelling']}")
    if "policy" in recipe:
        if not (isinstance(recipe["policy"], list) or isinstance(recipe["policy"], str)):
            raise ValueError(f"Invalid policy: {recipe['policy']}")
        if isinstance(recipe["policy"], str):
            recipe["policy"] = [recipe["policy"]]
    return recipe
