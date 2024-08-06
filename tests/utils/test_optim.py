import random as rnd
import time

import pytest

from fence.utils.base import time_it
from fence.utils.logger import setup_logging
from fence.utils.optim import parallelize, retry

logger = setup_logging(__name__)


# Test the retry decorator
def test_retry():
    @retry(max_retries=3, delay=1)
    def _test_retry():
        print("Testing function")
        raise Exception("This is a test exception")

    with pytest.raises(Exception, match="This is a test exception"):
        _test_retry()


# Test the time_it decorator
def test_time_it():
    @time_it(only_warn=False)
    def _test_time_it():
        print("Testing time_it")

    _test_time_it()


@parallelize
def helper_threaded_execution(item: str, other_kw_arg="test"):
    time.sleep(rnd.uniform(0, 3))
    logger.critical(
        f"Testing threaded_execution: thread {item} with kw arg {other_kw_arg}"
    )

    return item


def test_threaded_execution_decorator():
    results = helper_threaded_execution(
        ["this", "is", "a", "test"], other_kw_arg="test"
    )
    assert results == ["this", "is", "a", "test"]

    # class TestClass:
    #     def __init__(self):
    #         pass
    #
    #     @parallelize
    #     def test_threaded_execution(self, item: str, other_kw_arg="test"):
    #         time.sleep(rnd.uniform(0, 3))
    #         logger.critical(
    #             f"Testing threaded_execution: thread {item} with kw arg {other_kw_arg}"
    #         )
    #
    #         return item
    #
    # test_class = TestClass()
    # results = test_class.test_threaded_execution(
    #     ["this", "is", "a", "test"], other_kw_arg="test"
    # )

    assert results == ["this", "is", "a", "test"]


if __name__ == "__main__":
    pytest.main()
