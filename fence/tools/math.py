"""
Math based tools
"""

import numexpr

from fence import setup_logging

from .base import BaseTool

logger = setup_logging(__name__, log_level="info", serious_mode=False)


class PrimeTool(BaseTool):
    """Checks if a number is prime"""

    def run(self, number: int) -> bool:
        """Check if the number is prime"""
        logger.info(f"{self.__class__.__name__} was called: {number}")

        if number < 2:
            return False
        for i in range(2, number):
            if number % i == 0:
                return False
        return True


class CalculatorTool(BaseTool):
    """Perform mathematical calculations"""

    def run(self, expression: str) -> float | str:
        """Evaluate the mathematical expression"""

        logger.info(f"{self.__class__.__name__} was called: {expression}")
        try:

            result = numexpr.evaluate(expression)

        except Exception as e:
            logger.error(f"Error evaluating expression: {e}", exc_info=True)
            return f"Error evaluating expression {expression} - {e}"
        return result
