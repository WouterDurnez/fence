"""
Math based tools
"""

import logging
import os

from fence.tools.base import BaseTool

# Set up numexpr logging early, to suppress annoying logs
numexpr_logger = logging.getLogger("numexpr")
numexpr_logger.setLevel(logging.WARNING)
os.environ["NUMEXPR_MAX_THREADS"] = "4"
from numexpr.utils import set_num_threads  # noqa

set_num_threads(4)


logger = logging.getLogger(__name__)


class PrimeTool(BaseTool):
    """Checks if a number is prime"""

    def execute_tool(self, number: int, **kwargs) -> bool:
        """
        Check if the number is prime and optionally log based on environment variables.
        """
        logger.info(f"{self.__class__.__name__} was called with number: {number}")

        # Access environment variables passed to the run method
        log_check = kwargs.get("environment", {}).get("log_check", True)

        if log_check:
            logger.info(f"Checking if {number} is prime...")

        if number < 2:
            return False
        for i in range(2, number):
            if number % i == 0:
                return False
        return True


class CalculatorTool(BaseTool):
    """Perform mathematical calculations"""

    def execute_tool(self, expression: str, **kwargs) -> float | str:
        """
        Evaluate the mathematical expression.
        Optionally, environment can influence behavior, like precision or logging.
        """
        logger.info(
            f"{self.__class__.__name__} was called with expression: {expression}"
        )
        try:
            import numexpr

            result = numexpr.evaluate(expression)

            # Access environment variables directly from kwargs
            environment = kwargs.get("environment", {})

            # Optional precision from environment
            precision = environment.get("precision", None)
            if precision is not None and isinstance(result, (int, float)):
                result = round(result, precision)

            # Optionally adjust logging based on environment
            log_level = environment.get("log_level", "info")
            if log_level == "debug":
                logger.debug(f"Expression: {expression}, Result: {result}")
            else:
                logger.info(f"Expression: {expression}, Result: {result}")

        except Exception as e:
            logger.error(f"Error evaluating expression: {e}", exc_info=True)
            return f"Error evaluating expression {expression} - {e}"

        return result
