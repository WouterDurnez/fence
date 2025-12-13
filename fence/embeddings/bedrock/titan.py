"""
Titan embedding models for AWS Bedrock.

This module provides embedding models for:
- Titan Text Embeddings V2 (text-only)
- Titan Multimodal Embeddings G1 (text + image)
"""

import logging
from typing import Literal

from fence.embeddings.bedrock.base import (
    BedrockEmbeddingsBase,
    BedrockMultimodalEmbeddingsBase,
)

logger = logging.getLogger(__name__)

# Model IDs
MODEL_ID_AMAZON_TITAN_EMBED_TEXT_V2 = "amazon.titan-embed-text-v2:0"
MODEL_ID_AMAZON_TITAN_EMBED_IMAGE_V1 = "amazon.titan-embed-image-v1"

# Valid embedding dimensions for Titan models
TITAN_TEXT_V2_DIMENSIONS = (256, 512, 1024)  # V2 supports these
TITAN_MULTIMODAL_G1_DIMENSIONS = (256, 384, 1024)  # Multimodal G1 supports these


class TitanEmbeddingsV2(BedrockEmbeddingsBase):
    """
    Titan Text Embeddings V2 model.

    This model generates text embeddings with configurable dimensions.
    Max input: 8,192 tokens.

    Example:
        >>> titan = TitanEmbeddingsV2(source="search", dimensions=1024)
        >>> embedding = titan.embed("Hello world!")
    """

    def __init__(
        self,
        source: str | None = None,
        dimensions: Literal[256, 512, 1024] = 1024,
        normalize: bool = True,
        **kwargs
    ):
        """
        Initialize Titan Text Embeddings V2 model.

        :param str source: An indicator of where the model is operating from.
        :param int dimensions: Output embedding dimensions (256, 512, or 1024).
        :param bool normalize: Whether to normalize the embedding vectors.
        :param **kwargs: Additional arguments passed to BedrockEmbeddingsBase.
        """
        super().__init__(source=source, **kwargs)

        if dimensions not in TITAN_TEXT_V2_DIMENSIONS:
            raise ValueError(
                f"Invalid dimensions: {dimensions}. "
                f"Must be one of {TITAN_TEXT_V2_DIMENSIONS}"
            )

        self.model_id = MODEL_ID_AMAZON_TITAN_EMBED_TEXT_V2
        self.model_name = "Titan Text Embeddings V2"
        self.dimensions = dimensions
        self.normalize = normalize

    def _embed(self, text: str) -> dict:
        """
        Embed text using Titan Text Embeddings V2.

        :param text: Text to embed.
        :return: Response dictionary containing the embedding.
        """
        request_body = {
            "inputText": text,
            "dimensions": self.dimensions,
            "normalize": self.normalize,
        }
        return self._invoke_model(request_body)


class TitanMultimodalEmbeddingsG1(BedrockMultimodalEmbeddingsBase):
    """
    Titan Multimodal Embeddings G1 model.

    This model generates embeddings for text, images, or both together
    in the same semantic space. Useful for cross-modal search and similarity.

    - Max input text tokens: 256
    - Max image size: 25 MB
    - Supported formats: JPEG, PNG, GIF, WebP

    Example:
        >>> titan = TitanMultimodalEmbeddingsG1(source="search")
        >>> text_embedding = titan.embed("A red car")
        >>> image_embedding = titan.embed_image(image_bytes, image_format="jpeg")
        >>> combined = titan.embed_multimodal(text="A car", image=image_bytes)
    """

    def __init__(
        self,
        source: str | None = None,
        output_embedding_length: Literal[256, 384, 1024] = 1024,
        **kwargs
    ):
        """
        Initialize Titan Multimodal Embeddings G1 model.

        :param str source: An indicator of where the model is operating from.
        :param int output_embedding_length: Output embedding dimensions (256, 384, or 1024).
        :param **kwargs: Additional arguments passed to BedrockMultimodalEmbeddingsBase.
        """
        super().__init__(source=source, **kwargs)

        if output_embedding_length not in TITAN_MULTIMODAL_G1_DIMENSIONS:
            raise ValueError(
                f"Invalid output_embedding_length: {output_embedding_length}. "
                f"Must be one of {TITAN_MULTIMODAL_G1_DIMENSIONS}"
            )

        self.model_id = MODEL_ID_AMAZON_TITAN_EMBED_IMAGE_V1
        self.model_name = "Titan Multimodal Embeddings G1"
        self.output_embedding_length = output_embedding_length

    def _build_embedding_config(self) -> dict:
        """Build the embeddingConfig for the API request."""
        return {"outputEmbeddingLength": self.output_embedding_length}

    def embed(self, text: str) -> list[float]:
        """
        Embed text.

        :param text: Text to embed (max 256 tokens).
        :return: Embedding as a list of floats.
        """
        self._check_if_text_is_valid(text)

        request_body = {
            "inputText": text,
            "embeddingConfig": self._build_embedding_config(),
        }
        response = self._invoke_model(request_body)
        embedding = response["embedding"]

        return response if self.full_response else embedding

    def embed_image(
        self,
        image: bytes | str,
        image_format: str | None = None  # noqa: ARG002 - kept for interface compatibility
    ) -> list[float]:
        """
        Embed an image.

        :param image: Image as bytes or base64-encoded string.
        :param image_format: Image format (optional, not used by Titan but kept for interface).
        :return: Embedding as a list of floats.
        """
        _ = image_format  # Titan doesn't require format specification
        image_base64 = self._encode_image_to_base64(image)

        request_body = {
            "inputImage": image_base64,
            "embeddingConfig": self._build_embedding_config(),
        }
        response = self._invoke_model(request_body)
        embedding = response["embedding"]

        return response if self.full_response else embedding

    def embed_multimodal(
        self,
        text: str | None = None,
        image: bytes | str | None = None,
        image_format: str | None = None  # noqa: ARG002 - kept for interface compatibility
    ) -> list[float]:
        """
        Embed text and/or image together.

        :param text: Optional text to embed.
        :param image: Optional image as bytes or base64-encoded string.
        :param image_format: Optional image format (not used by Titan).
        :return: Embedding as a list of floats.
        :raises ValueError: If neither text nor image is provided.
        """
        _ = image_format  # Titan doesn't require format specification
        if text is None and image is None:
            raise ValueError("At least one of 'text' or 'image' must be provided")

        request_body = {"embeddingConfig": self._build_embedding_config()}

        if text is not None:
            self._check_if_text_is_valid(text)
            request_body["inputText"] = text

        if image is not None:
            request_body["inputImage"] = self._encode_image_to_base64(image)

        response = self._invoke_model(request_body)
        embedding = response["embedding"]

        return response if self.full_response else embedding


if __name__ == "__main__":
    # Example usage - Text Embeddings V2
    text = "The quick brown fox jumps over the lazy dog"
    titan_text = TitanEmbeddingsV2(source="example", dimensions=1024, full_response=True)
    embedding = titan_text.embed(text)
    print(f"Text V2 embedding: {embedding}")

    # Example usage - Multimodal G1 (text only)
    titan_mm = TitanMultimodalEmbeddingsG1(source="example", output_embedding_length=1024)
    text_embedding = titan_mm.embed("A beautiful sunset")
    print(f"Multimodal text embedding dimension: {len(text_embedding)}")