"""
Optimization utils
"""

import functools
import inspect
import logging
import queue
import time
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Callable, Iterable

logger = logging.getLogger(__name__)


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


def parallelize(
    f: Callable = None, max_workers: int = 4, raise_exceptions: bool = False
):
    """Decorator that parallelizes function execution across multiple inputs.

    :param f: Function to be parallelized.
    :param max_workers: Maximum number of worker threads, defaults to 4.
    :param raise_exceptions: If True, raises the first encountered exception. If False, logs exceptions and returns successful results.
    :return: Wrapped function that performs parallel execution.
    """

    if f is None:
        return lambda func: parallelize(
            func, max_workers=max_workers, raise_exceptions=raise_exceptions
        )

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        results_queue = queue.Queue()

        def run_func(index, *args, **kwargs):
            """Execute the original function and handle exceptions.

            :param index: Index of the current execution for ordering results.
            """
            logger.debug(
                f"[Thread {index}] Running function {f.__name__} with {args=}, {kwargs=}"
            )
            try:
                result = f(*args, **kwargs)
                results_queue.put((index, result))
                logger.debug(f"[Thread {index}] Function {f.__name__} completed")
            except Exception as e:
                logger.error(f"[Thread {index}] Error in function {f.__name__}: {e}")
                results_queue.put((index, e))

        # Handle empty input case
        if not args:
            return []

        # Check if the function is a method (i.e., first argument is 'self')
        is_method = inspect.ismethod(f) or (
            len(args) > 1 and not isinstance(args[0], Iterable)
        )

        if is_method:
            self_arg = args[0]  # Store the 'self' argument
            args_to_parallelize = args[1]  # The iterable of argument tuples
        else:
            self_arg = None
            args_to_parallelize = args[
                0
            ]  # For standalone functions, use the first argument

        # Handle single input case
        if not isinstance(args_to_parallelize, Iterable) or isinstance(
            args_to_parallelize, str
        ):
            args_to_parallelize = [args_to_parallelize]

        # Use a ThreadPoolExecutor to run the functions in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for index, parallel_args in enumerate(args_to_parallelize):
                # Ensure parallel_args is a tuple, even if it's a single item
                if not isinstance(parallel_args, tuple):
                    parallel_args = (parallel_args,)

                # Prepend 'self' for methods
                if self_arg is not None:
                    full_args = (self_arg,) + parallel_args
                else:
                    full_args = parallel_args

                # Submit the task to the executor
                futures.append(executor.submit(run_func, index, *full_args, **kwargs))

        # Wait for all tasks to complete
        wait(futures)

        # Collect and order the results
        results = list(results_queue.queue)
        results = [result[1] for result in sorted(results, key=lambda x: x[0])]

        # Filter out exceptions and log them
        successful_results = []
        exceptions = []
        for result in results:
            if isinstance(result, Exception):
                exceptions.append(result)
            else:
                successful_results.append(result)

        if exceptions:
            logger.warning(
                f"Encountered {len(exceptions)} errors during parallel execution"
            )
            for e in exceptions[:5]:  # Log the first 5 exceptions
                logger.warning(f"Error encountered: {str(e)}")

            if raise_exceptions:
                raise exceptions[0]

        return successful_results

    return wrapper
