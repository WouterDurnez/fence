"""
Various utils
"""

import functools
import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path

CONF_DIR = Path(__file__).resolve().parent.parent.parent / "conf"
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

logger = logging.getLogger(__name__)


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
            logger.warning(message)
        elif not only_warn:
            logger.info(message)
        else:
            pass

        return res

    return timed


def time_out(
    f=None, seconds: int = 60, raise_exception: bool = True, default_return=None
):
    """Decorator that adds a timeout to a function.

    :param f: Function to be executed.
    :param seconds: Timeout in seconds.
    :param raise_exception: If True, raises an exception if the timeout is exceeded.
    :param default_return: Default return value if the timeout is exceeded.
    :return: Wrapped function with a timeout.
    """

    if f is None:
        return lambda func: time_out(
            func,
            seconds=seconds,
            raise_exception=raise_exception,
            default_return=default_return,
        )

    @functools.wraps(f)
    def wrapper_timeout(*args, **kwargs):
        """Wrapper function for the timeout decorator."""

        def run_func(*args, **kwargs):
            """Execute the original function and handle exceptions."""
            try:
                return f(*args, **kwargs)
            except Exception as e:
                raise e

        # Use a ThreadPoolExecutor to run the function in a separate thread
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_func, *args, **kwargs)

            try:
                return future.result(timeout=seconds)
            except TimeoutError:

                # Log a warning or raise an exception if the timeout is exceeded
                if raise_exception:
                    raise TimeoutError(f"Function call <{f.__name__}> timed out")
                else:
                    logger.warning(f"Function call <{f.__name__}> timed out")
                    return default_return

    return wrapper_timeout


def setup_demo():

    import sys
    from pathlib import Path

    parent_dir = Path(__file__).resolve().parents[2]
    print(parent_dir)
    sys.path.append(parent_dir)
