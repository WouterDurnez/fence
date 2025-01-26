"""
Claude Gen 3 models
"""

import logging

from fence.models import BedrockBase
from fence.models.base import register_log_callback, register_log_tags
from fence.utils.logger import setup_logging

MODEL_ID_PRO = "amazon.nova-pro-v1:0"
MODEL_ID_LITE = "amazon.nova-lite-v1:0"
MODEL_ID_MICRO = "amazon.nova-micro-v1:0"


logger = logging.getLogger(__name__)


class NovaPro(BedrockBase):
    """Amazon Nova Pro model class"""

    def __init__(self, **kwargs):
        """
        Initialize a Claude Haiku model
        :param str source: An indicator of where (e.g., which feature) the model is operating from.
        :param **kwargs: Additional keyword arguments
        """

        super().__init__(**kwargs)

        self.model_id = MODEL_ID_PRO
        self.model_name = "Nova Pro"


class NovaLite(BedrockBase):
    """Amazon Nova Lite model class"""

    def __init__(self, **kwargs):
        """
        Initialize a Claude Haiku model
        :param str source: An indicator of where (e.g., which feature) the model is operating from.
        :param **kwargs: Additional keyword arguments
        """

        super().__init__(**kwargs)

        self.model_id = MODEL_ID_LITE
        self.model_name = "Nova Lite"


class NovaMicro(BedrockBase):
    """Amazon Nova Micro model class"""

    def __init__(self, **kwargs):
        """
        Initialize a Claude Haiku model
        :param str source: An indicator of where (e.g., which feature) the model is operating from.
        :param **kwargs: Additional keyword arguments
        """

        super().__init__(**kwargs)

        self.model_id = MODEL_ID_MICRO
        self.model_name = "Nova Micro"


if __name__ == "__main__":

    setup_logging(log_level="debug", are_you_serious=False)

    # Register logging callback
    register_log_callback(lambda metrics, tags: print(metrics, tags))

    # Register logging tags
    register_log_tags({"team": "data-science-test", "project": "fence"})

    # Create an instance of the ClaudeHaiku class
    model = NovaPro(source="test", metric_prefix="yolo", region="us-east-1")

    # Call the invoke method with a prompt
    response = model.invoke(prompt="The sun is shining brightly")
