from fence import setup_logging

from .base import BaseTool

logger = setup_logging(__name__, log_level="info", serious_mode=False)


class TextInverterTool(BaseTool):
    """Inverts the text"""

    def run(self, text: str) -> str:
        """Invert the text"""
        logger.info(f"{self.__class__.__name__} was called: {text}")

        return text[::-1]
