"""
Various utils
"""

import logging
import time
from pathlib import Path

from fence.utils.logger import setup_logging

CONF_DIR = Path(__file__).resolve().parent.parent.parent / "conf"
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

logger = setup_logging(__name__)


def time_it(f=None, threshold: int = 300, only_warn: bool = True):
    """
    Timer decorator: shows how long execution of function took.
    :param f: function to measure
    :param threshold: threshold for warning
    :param only_warn: whether to only log a warning if the threshold is exceeded
    """
    # Check if the decorator is used without parentheses
    if f is None:
        return lambda func: time_it(func, threshold=threshold, only_warn=only_warn)

    setup_logging()
    log = logging.getLogger(__name__)

    def timed(*args, **kwargs):
        t1 = time.time()
        res = f(*args, **kwargs)
        t2 = time.time()
        duration = round(t2 - t1, 2)

        # If the first argument is an object, get its class name
        class_name, object_name = None, None
        if args:
            class_name = (
                args[0].__class__.__name__
                if hasattr(args[0], "__class__") and not isinstance(args[0], dict)
                else None
            )
            object_name = args[0].name if hasattr(args[0], "name") else None
        method_name = f.__name__
        message = f"Method <{class_name if class_name else ''}{(f'[{object_name}]' if object_name else '')}{'.' if any([class_name, object_name]) else ''}{method_name}> took {duration} s to execute."
        if duration > threshold:
            log.warning(message)
        elif not only_warn:
            log.info(message)
        else:
            pass

        return res

    return timed
