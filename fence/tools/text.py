from fence import setup_logging

from .base import BaseTool

logger = setup_logging(__name__, log_level="info", serious_mode=False)


class TextInverterTool(BaseTool):
    """Inverts the text"""

    def execute_tool(self, text: str, **kwargs) -> str:
        """Invert the text"""
        logger.info(f"{self.__class__.__name__} was called: {text}")

        return text[::-1]


# class CalculatorTool(BaseTool):
#     """Perform mathematical calculations"""
#
#     def execute_tool(self, expression: str, **kwargs) -> float | str:
#         """
#         Evaluate the mathematical expression.
#         Optionally, environment can influence behavior, like precision or logging.
#         """
#         logger.info(f"{self.__class__.__name__} was called with expression: {expression}")
#         try:
#             import numexpr
#
#             result = numexpr.evaluate(expression)
#
#             # Access environment variables directly from kwargs
#             environment = kwargs.get('environment', {})
#
#             # Optional precision from environment
#             precision = environment.get('precision', None)
#             if precision is not None and isinstance(result, (int, float)):
#                 result = round(result, precision)
#
#             # Optionally adjust logging based on environment
#             log_level = environment.get('log_level', 'info')
#             if log_level == 'debug':
#                 logger.debug(f"Expression: {expression}, Result: {result}")
#             else:
#                 logger.info(f"Expression: {expression}, Result: {result}")
#
#         except Exception as e:
#             logger.error(f"Error evaluating expression: {e}", exc_info=True)
#             return f"Error evaluating expression {expression} - {e}"
#
#         return result
