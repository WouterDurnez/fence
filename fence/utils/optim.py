"""
Optimization utils
"""

import functools
import queue
import random as rnd
import time
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Callable, Iterable

from fence.utils.logging import setup_logging

logger = setup_logging(__name__)


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
            final_error = None
            while retries < max_retries:
                try:
                    return f(
                        *args, **kwargs
                    )  # Pass received args and kwargs to the original function
                except Exception as e:
                    retries += 1
                    error_message = (
                        f"Error in {f.__name__}, attempt {retries}/{max_retries}: {e}"
                    )
                    logger.warning(error_message)
                    final_error = e
                    time.sleep(delay)
            logger.warning(f"Maximum retries reached for {f.__name__}")
            raise RuntimeError(
                f"Maximum retries reached for {f.__name__}. Final error: {final_error}"
            )

        return decorated_function(
            *args, **kwargs
        )  # Call the decorated function with the received args and kwargs

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
            result = f(*args, **kwargs)

            # Put the result in the queue, along with the index to keep track of the order
            results_queue.put((index, result))
            logger.debug(f"[Thread {index}] Function {f.__name__} completed")

        # Check if first argument is an iterable, if not, split the args into a list
        first_arg_iterable = isinstance(args[0], Iterable)
        if not first_arg_iterable:
            self_arg = args[0]
            args = args[1:]
            logger.debug(
                f"First argument is not an iterable. Using {self_arg} as self."
            )
        else:
            self_arg = None
            logger.debug("First argument is an iterable. Not using self.")

        # Use ThreadPoolExecutor to run the functions in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            if not first_arg_iterable:
                futures = [
                    executor.submit(run_func, index, self_arg, *args, **kwargs)
                    for index, args in enumerate(zip(*args))
                ]
            else:
                futures = [
                    executor.submit(run_func, index, *args, **kwargs)
                    for index, args in enumerate(zip(*args))
                ]

        # Wait for all tasks to complete
        wait(futures)

        # Convert the queue to a list
        results = list(results_queue.queue)

        # Sort the results by index and remove the index
        results = [result[1] for result in sorted(results, key=lambda x: x[0])]

        return results

    return wrapper


@parallelize
def test_threaded_execution(item: str, other_kw_arg="test"):
    time.sleep(rnd.uniform(0, 3))
    logger.critical(
        f"Testing threaded_execution: thread {item} with kw arg {other_kw_arg}"
    )

    return item


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
    # @time_it(only_warn=False)
    # def test_time_it():
    #     print("Testing time_it")
    #
    #
    # test_time_it()

    # Test the threaded_execution decorator
    import random as rnd

    results = test_threaded_execution(["this", "is", "a", "test"], other_kw_arg="test")

    # Test with a class method
    class TestClass:
        def __init__(self):
            pass

        @parallelize
        def test_threaded_execution(self, item: str, other_kw_arg="test"):
            time.sleep(rnd.uniform(0, 3))
            logger.critical(
                f"Testing threaded_execution: thread {item} with kw arg {other_kw_arg}"
            )

            return item

    test_class = TestClass()
    results = test_class.test_threaded_execution(
        ["this", "is", "a", "test"], other_kw_arg="test"
    )
