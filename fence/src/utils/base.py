import functools
import logging
import os
import queue
import time
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv
from rich.logging import Console, RichHandler

load_dotenv()

CONF_DIR = Path(__file__).resolve().parent.parent.parent / "conf"
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

LOGGING_FORMAT = "%(message)s"
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
LOG_LEVEL = getattr(logging, LOG_LEVEL)


def setup_logging(log_level: str = LOG_LEVEL):
    """
    Utility function to set up logging in various parts of the code.
    """
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


setup_logging()
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


################
# Optimization #
################


def retry(f=None, max_retries=3, delay=0.2):
    """
    Retry decorator: retries a function if it fails.
    :param max_retries: maximum number of retries
    :param delay: delay between retries
    """

    # Check if the decorator is used without parentheses
    if f is None or not callable(f):
        # ...if so, return a lambda function that takes the function as an argument
        return lambda func: retry(func, max_retries=max_retries, delay=delay)

    @functools.wraps(f)
    def wrapper_retry(*args, **kwargs):
        """
        Wrapper function for the retry decorator. We need this to pass the function to the decorator.
        """

        def decorated_function(*args, **kwargs):
            """
            Decorated function that retries the function if it fails.
            """
            retries = 0
            while retries < max_retries:
                try:
                    return f(*args, **kwargs)  # Pass received args and kwargs to the original function
                except Exception as e:
                    retries += 1
                    error_message = f"Error in {f.__name__}, attempt {retries}/{max_retries}: {e}"
                    logger.warning(error_message)
                    time.sleep(delay)
            raise RuntimeError(f"Maximum retries reached for {f.__name__}")

        return decorated_function(*args, **kwargs)  # Call the decorated function with the received args and kwargs

    return wrapper_retry


def parallelize(f: Callable = None, max_workers: int = 4):
    """
    Parallelize decorator: runs a function in parallel.
    :param f: function to run in parallel
    :param max_workers: maximum number of workers
    """

    # Check if the decorator is used without parentheses
    if f is None:

        # ...if so, return a lambda function that takes the function as an argument
        return lambda func: parallelize(func, max_workers=max_workers)

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        """
        Wrapper function for the parallelize decorator. We need this to pass the function's arguments to the decorator.
        """
        # Create a queue to store the results. We do not use a list because doing so is, in principle, not thread-safe.
        # It probably could be, due to the GIL, but it's better to be safe than sorry.
        results_queue = queue.Queue()

        # Define the function to run in parallel
        def run_func(index, *args, **kwargs):
            logger.debug(f"[Thread {index}] Running function {f.__name__}")
            result = f(index, *args, **kwargs)
            results_queue.put(result)
            logger.debug(f"[Thread {index}] Function {f.__name__} completed")

        # Use ThreadPoolExecutor to run the functions in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(run_func, index, *args, **kwargs)
                for index, args in enumerate(zip(*args))
            ]

        # Wait for all tasks to complete
        wait(futures)

        # Convert the queue to a list
        results = list(results_queue.queue)

        return results

    return wrapper


if __name__ == "__main__":


    # @retry(max_retries=3, delay=1)
    # def test_retry():
    #     print("Testing function")
    #     raise Exception("This is a test exception")
    #
    #
    # try:
    #     test_retry()
    # except Exception as e:
    #     print("Final exception: ", e)


    # Test the time_it decorator
    @time_it(only_warn=False)
    def test_time_it():
        print("Testing time_it")


    test_time_it()


    # # Test the threaded_execution decorator
    # @parallelize
    # def test_threaded_execution(index: int, item: str):
    #     logger.critical(f"Testing threaded_execution: thread {item}")
    #
    #
    # test_threaded_execution(zip([1, 2, 3, 4], ["this", "is", "a", "test"]))
