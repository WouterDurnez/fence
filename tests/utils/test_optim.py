"""
Optimization utilities tests
"""

import random
import random as rnd
import time
from unittest.mock import patch

import pytest

from fence.utils.logger import setup_logging
from fence.utils.optim import parallelize, retry

logger = setup_logging(__name__, log_level="DEBUG")


def mock_api_call(id):
    """Simulate an API call with random delay and response."""
    time.sleep(rnd.uniform(0.1, 0.5))  # Simulate network delay
    return {"id": id, "data": f"Result for {id}"}


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

        @parallelize(raise_exceptions=True)
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

    def test_with_extra_kwargs(self):
        """Test using the decorator with extra kwargs."""

        @parallelize(max_workers=2)
        def add(x, y, z=0):
            return x + y + z

        args = [(1, 2), (3, 4), (5, 6)]
        result = add(args, z=1)
        assert result == [4, 8, 12]

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

    @patch("test_optim.mock_api_call", side_effect=mock_api_call)
    def test_parallelize_api_calls(self, mock_api):
        """Test parallelizing API calls."""

        mock_api.reset_mock()  # Reset the mock at the start of the test

        @parallelize(max_workers=5)
        def fetch_data(id):
            result = mock_api_call(id)
            logger.debug(f"Called with id: {id}, result: {result}")
            return result

        ids = list(range(10))
        results = fetch_data(ids)

        logger.debug(f"Number of mock calls: {mock_api.call_count}")
        logger.debug(f"Call arguments: {mock_api.call_args_list}")

        assert len(results) == 10, f"Should have 10 results, but got {len(results)}"
        assert all(r["id"] in ids for r in results), "All IDs should be in the results"
        assert all(
            r["data"].startswith("Result for") for r in results
        ), "All results should have correct data format"

        # Check if the mock was called the correct number of times
        assert (
            mock_api.call_count == 10
        ), f"API should be called 10 times, but was called {mock_api.call_count} times"

        # Log all the results
        logger.debug(f"Results: {results}")

    @patch("test_optim.mock_api_call", side_effect=mock_api_call)
    def test_parallelize_api_calls_with_errors(self, mock_api):
        """Test parallelizing API calls with some errors."""
        mock_api.reset_mock()  # Reset the mock at the start of the test

        def api_with_errors(id):
            if id % 3 == 0:
                logger.debug(f"Raising error for id: {id}")
                raise ValueError(f"Error for id {id}")
            result = mock_api_call(id)
            logger.debug(f"Called with id: {id}, result: {result}")
            return result

        @parallelize(max_workers=5)
        def fetch_data_with_errors(id):
            return api_with_errors(id)

        ids = list(range(10))

        try:
            results = fetch_data_with_errors(ids)  # noqa
        except ValueError as e:
            logger.debug(f"Caught ValueError: {str(e)}")

        logger.debug(f"Number of mock calls: {mock_api.call_count}")
        logger.debug(f"Call arguments: {mock_api.call_args_list}")

        # Check if the mock was called the correct number of times
        # Should not be called for ids 0, 3, 6, 9, so should be called 6 times
        assert (
            mock_api.call_count == 6
        ), f"API should be called 7 times, but was called {mock_api.call_count} times"

        # Log all the call arguments
        for call in mock_api.call_args_list:
            logger.debug(f"Call argument: {call}")

    @patch("test_optim.mock_api_call", side_effect=mock_api_call)
    def test_parallelize_api_calls_at_scale(self, mock_api):
        """Test parallelizing API calls at scale with timeout."""

        @parallelize(max_workers=20)
        def fetch_data(id):
            time.sleep(
                random.uniform(0.01, 0.05)
            )  # Simulate variable API response time
            if random.random() < 0.05:  # 5% chance of failure
                raise ValueError(f"Random error for id {id}")
            return {"id": id, "data": f"Result for {id}"}

        num_calls = 1000
        ids = list(range(num_calls))

        logger.debug(f"Starting scale test with {num_calls} calls")
        start_time = time.time()

        try:
            results = fetch_data(ids)
        except Exception as e:
            logger.error(f"Unexpected error during parallel execution: {str(e)}")

        end_time = time.time()
        parallel_time = end_time - start_time

        logger.debug(f"Parallel execution completed in {parallel_time:.2f} seconds")
        logger.debug(f"Number of successful calls: {len(results)}")

        assert (
            len(results) < num_calls
        ), "Some calls should have failed due to random errors"
        assert (
            parallel_time < 60
        ), f"Parallel execution took too long: {parallel_time:.2f} seconds"

        # Verify results
        for result in results:
            assert isinstance(result, dict), f"Expected dict, got {type(result)}"
            assert "id" in result, "Result should have 'id' key"
            assert "data" in result, "Result should have 'data' key"
            assert result["data"].startswith(
                "Result for"
            ), f"Unexpected data format: {result['data']}"

        logger.debug(f"All assertions passed for scale test with {num_calls} calls")

    class TestRetryDecorator:
        """Test the retry decorator"""

        def test_retry(self):
            """Test the retry decorator."""

            @retry(max_retries=3, delay=1)
            def _test_retry():
                print("Testing function")
                raise Exception("This is a test exception")

            with pytest.raises(Exception, match="This is a test exception"):
                _test_retry()


if __name__ == "__main__":
    pytest.main()
