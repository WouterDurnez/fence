import random as rnd
import time

import pytest

from fence.utils.base import time_it
from fence.utils.logger import setup_logging
from fence.utils.optim import parallelize, retry

logger = setup_logging(__name__)


class DummyClass:
    @parallelize
    def multiply(self, x, y):
        return x * y


class TestClass:
    @parallelize
    def threaded_execution(self, item: str, other_kw_arg="test"):
        time.sleep(rnd.uniform(0, 3))
        logger.critical(
            f"Testing threaded_execution: thread {item} with kw arg {other_kw_arg}"
        )
        return item


class TestParallelizeDecorator:

    @pytest.mark.parametrize(
        "decorator_args, input_args, expected",
        [
            (None, [(1, 2), (3, 4), (5, 6)], [3, 7, 11]),
            ({"max_workers": 2}, [(1, 2), (3, 4), (5, 6)], [3, 7, 11]),
        ],
    )
    def test_basic_functionality(self, decorator_args, input_args, expected):
        """Test the basic functionality of the parallelize decorator."""

        @parallelize(**(decorator_args or {}))
        def add(x, y):
            return x + y

        result = add(input_args)
        assert result == expected

    def test_order_preservation(self):
        """Ensure that the order of results is preserved."""

        @parallelize
        def slow_add(x, y):
            time.sleep(0.1)
            return x + y

        args = [(1, 2), (3, 4), (5, 6)]
        result = slow_add(args)
        assert result == [3, 7, 11]

    def test_scaling_with_workers(self):
        """Test scaling with different numbers of workers."""

        @parallelize(max_workers=2)
        def multiply(x, y):
            return x * y

        args = [(1, 2), (3, 4), (5, 6)]
        result = multiply(args)
        assert result == [2, 12, 30]

    def test_non_iterable_first_argument(self):
        """Test with non-iterable first argument (e.g., class methods)."""
        obj = DummyClass()
        args = [(2, 2), (3, 3), (4, 4)]
        result = obj.multiply(args)
        assert result == [4, 9, 16]

    def test_error_handling(self):
        """Test error handling in parallelized functions."""

        @parallelize
        def raise_exception(x):
            raise ValueError("Test Error")

        args = [(1,), (2,), (3,)]
        with pytest.raises(ValueError, match="Test Error"):
            raise_exception(args)

    def test_edge_cases(self):
        """Test various edge cases."""

        @parallelize
        def square(x):
            return x * x

        assert square() == [], "Empty input should return an empty list"
        assert square([4]) == [16], "Single input should work correctly"

        large_input = list(range(1000))
        assert square(large_input) == [
            x * x for x in large_input
        ], "Large input should be processed correctly"

    def test_performance(self):
        """Test performance considerations."""

        @parallelize
        def add(x, y):
            return x + y

        large_input = [(i, i + 1) for i in range(1000)]

        start_time = time.time()
        result = add(large_input)
        end_time = time.time()

        assert result == [
            x + y for x, y in large_input
        ], "Results should be correct for large input"
        assert (
            end_time - start_time < 2
        ), "Parallelized version should be faster than a certain threshold"

    def test_decorator_without_parentheses(self):
        """Test using the decorator without parentheses."""

        @parallelize
        def subtract(x, y):
            return x - y

        args = [(5, 2), (8, 3), (10, 7)]
        result = subtract(args)
        assert result == [3, 5, 3]

    def test_class_method(self):
        """Test parallelized class method."""
        test_instance = TestClass()
        items = ["one", "two", "three", "four"]
        results = test_instance.threaded_execution(items, other_kw_arg="test")
        assert set(results) == set(items), "The results do not match the expected items"

    def test_more_workers_than_available(self):
        """Test using more workers than available."""

        @parallelize(max_workers=1000)  # Much more than typically available
        def simple_func(x):
            return x * 2

        inputs = list(range(1000))
        results = simple_func(inputs)
        assert results == [
            x * 2 for x in inputs
        ], "Results should be correct even with more workers than available"


class TestOtherDecorators:

    def test_retry(self):
        """Test the retry decorator."""

        @retry(max_retries=3, delay=1)
        def _test_retry():
            print("Testing function")
            raise Exception("This is a test exception")

        with pytest.raises(Exception, match="This is a test exception"):
            _test_retry()

    def test_time_it(self):
        """Test the time_it decorator."""

        @time_it(only_warn=False)
        def _test_time_it():
            print("Testing time_it")

        _test_time_it()


if __name__ == "__main__":
    pytest.main()
