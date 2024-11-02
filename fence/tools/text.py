import logging

from .base import BaseTool

logger = logging.getLogger(__name__)


class TextInverterTool(BaseTool):
    """Inverts the text"""

    def execute_tool(self, text: str, **kwargs) -> str:
        """Invert the text"""
        logger.info(f"{self.__class__.__name__} was called: {text}")

        return text[::-1]
