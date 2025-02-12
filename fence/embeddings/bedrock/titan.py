"""
Titan embedding models
"""

import logging

from fence.embeddings.bedrock.base import BedrockEmbeddingsBase

logger = logging.getLogger(__name__)

MODEL_ID_AMAZON_TITAN_EMBED_TEXT_V2 = "amazon.titan-embed-text-v2:0"


class TitanEmbeddingsV2(BedrockEmbeddingsBase):
    """Titan Text Embeddings V2 model class"""

    def __init__(self, **kwargs):
        """
        Initialize a Titan Text Embeddings V2 model
        """

        super().__init__(**kwargs)

        self.model_id = MODEL_ID_AMAZON_TITAN_EMBED_TEXT_V2
        self.model_name = "Titan Text Embeddings V2"
