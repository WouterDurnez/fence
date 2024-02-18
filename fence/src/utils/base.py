import functools
import logging
import os
import queue
import time
from concurrent.futures import ThreadPoolExecutor, wait
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
        return lambda func: time_it(func, threshold=threshold)

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
        message = f"Method <{class_name if class_name else ''}{(f'[{object_name}]' if object_name else '')}{'.' if any([class_name,object_name]) else ''}{method_name}> took {duration} s to execute."
        if duration > threshold:
            log.warning(message)
        elif not only_warn:
            log.info(message)
        else:
            pass

        return res

    return timed


def retry(max_retries: int = 3, delay: float = 0.2):
    """
    Retry decorator: retries a function if it fails.
    :param max_retries: maximum number of retries
    :param delay: delay between retries
    """

    def wrapper_retry(func):
        """
        Wrapper function for the retry decorator. We need this to pass the function to the decorator.
        """

        @functools.wraps(func)  # This is necessary to preserve the function's metadata
        def decorated_function(*args, **kwargs):
            """
            Decorated function that retries the function if it fails.
            """
            # Start the retries
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    error_message = f"Error in {func.__name__}, attempt {retries}/{max_retries}: {e}"
                    logger.warning(error_message)
                    time.sleep(delay)  # Adjust the sleep duration between retries
            raise RuntimeError(f"Maximum retries reached for {func.__name__}")

        return decorated_function

    return wrapper_retry


def parallelize(max_workers: int = 4):
    """
    Parallelize decorator: runs a function in parallel.
    :param max_workers: maximum number of workers
    """

    def decorator(func):
        """
        Decorator function for the parallelize decorator. We need this to pass the function to the decorator.
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            """
            Wrapper function for the parallelize decorator. We need this to pass the function's arguments to the decorator.
            """
            # Create a queue to store the results. We do not use a list because doing so is, in principle, not thread-safe.
            # It probably could be, due to the GIL, but it's better to be safe than sorry.
            results_queue = queue.Queue()

            # Define the function to run in parallel
            def run_func(index, *args, **kwargs):
                logger.debug(f"[Thread {index}] Running function {func.__name__}")
                result = func(index, *args, **kwargs)
                results_queue.put(result)
                logger.debug(f"[Thread {index}] Function {func.__name__} completed")

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

    return decorator


if __name__ == "__main__":

    def test_retry():
        @retry(max_retries=3, delay=0.2)
        def test_function():
            print("Testing function")
            raise Exception("This is a test exception")

        test_function()

    try:
        test_retry()
    except Exception as e:
        print("Final exception: ", e)

    # Test the time_it decorator
    @time_it
    def test_time_it():
        print("Testing time_it")
        time.sleep(1)

    test_time_it()

    # Test the threaded_execution decorator
    @threaded_execution
    def test_threaded_execution(iterable: list[int]):
        logger.critical(f"Testing threaded_execution: thread {iterable}")

    test_threaded_execution([1, 2, 3, 4])
