"""
Base utilities tests
"""

import time

import pytest

from fence.utils.base import time_it, time_out


class TestTimeDecorators:

    def test_time_it(self):
        """Test the time_it decorator."""

        @time_it(only_warn=False)
        def _test_time_it():
            print("Testing time_it")

        _test_time_it()

    def test_time_out(self):
        """Test the time_out decorator."""

        @time_out(seconds=1)
        def _test_time_out():
            print("Testing time_out")
            time.sleep(2)

        with pytest.raises(Exception, match="Function call <_test_time_out> timed out"):
            _test_time_out()

    def test_timeout_default_value(self):
        """Test the time_out decorator with default value."""

        @time_out(seconds=1, raise_exception=False, default_return="Timeout")
        def _test_time_out_default():
            print("Testing time_out with default value")
            time.sleep(2)

        assert _test_time_out_default() == "Timeout"
