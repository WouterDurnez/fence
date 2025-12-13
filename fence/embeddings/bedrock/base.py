"""
Base class for Bedrock embeddings
"""

import base64
import json
import logging
from typing import Literal

import boto3

from fence.embeddings.base import Embeddings, MultimodalEmbeddings


logger = logging.getLogger(__name__)


class BedrockEmbeddingsBase(Embeddings):
    """Base class for Bedrock text embeddings"""

    inference_type = "bedrock"

    def __init__(
        self,
        source: str | None = None,
        full_response: bool = False,
        region: str = "eu-central-1",
        **kwargs
    ):
        """
        Initialize a Bedrock embeddings model

        :param str source: An indicator of where (e.g., which feature) the model is operating from.
        :param bool full_response: Whether to return the full response object or just the embedding.
        :param str region: AWS region for the Bedrock service.
        :param **kwargs: Additional keyword arguments
        """

        super().__init__(source=source)

        self.full_response = full_response
        self.region = region

        # Initialize the client
        self.client = boto3.client("bedrock-runtime", region_name=self.region)

    def embed(self, text: str) -> list[float]:

        self._check_if_text_is_valid(text=text)

        response = self._embed(text)

        embedding = response["embedding"]

        # Depending on the full_response flag, return either the full response or just the embedding
        return response if self.full_response else embedding

    def _embed(self, text: str) -> dict:
        """
        Embed query text. Override this in subclasses for specific model implementations.

        :param text: Text to embed.
        :return: Response dictionary containing the embedding.
        """
        raise NotImplementedError("Subclasses must implement _embed method")

    def _invoke_model(self, request_body: dict) -> dict:
        """
        Invoke the Bedrock model with the given request body.

        :param request_body: The request body to send to the model.
        :return: The model response as a dictionary.
        """
        try:
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body)
            )
            return json.loads(response["body"].read())
        except Exception as e:
            raise ValueError(f"Error raised by Bedrock service: {e}")

    @staticmethod
    def _check_if_text_is_valid(text: str):
        """Check if the text is a valid non-empty string"""
        if isinstance(text, str) and not text.strip():
            raise ValueError("Text cannot be empty string!")


class BedrockMultimodalEmbeddingsBase(MultimodalEmbeddings):
    """Base class for Bedrock multimodal embeddings (text + image)"""

    inference_type = "bedrock"

    # Valid image formats for multimodal embeddings
    VALID_IMAGE_FORMATS = {"jpeg", "jpg", "png", "gif", "webp"}

    def __init__(
        self,
        source: str | None = None,
        full_response: bool = False,
        region: str = "eu-central-1",
        **kwargs
    ):
        """
        Initialize a Bedrock multimodal embeddings model

        :param str source: An indicator of where (e.g., which feature) the model is operating from.
        :param bool full_response: Whether to return the full response object or just the embedding.
        :param str region: AWS region for the Bedrock service.
        :param **kwargs: Additional keyword arguments
        """

        super().__init__(source=source)

        self.full_response = full_response
        self.region = region

        # Initialize the client
        self.client = boto3.client("bedrock-runtime", region_name=self.region)

    def _invoke_model(self, request_body: dict) -> dict:
        """
        Invoke the Bedrock model with the given request body.

        :param request_body: The request body to send to the model.
        :return: The model response as a dictionary.
        """
        try:
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body)
            )
            return json.loads(response["body"].read())
        except Exception as e:
            raise ValueError(f"Error raised by Bedrock service: {e}")

    @staticmethod
    def _check_if_text_is_valid(text: str):
        """Check if the text is a valid non-empty string"""
        if isinstance(text, str) and not text.strip():
            raise ValueError("Text cannot be empty string!")

    @staticmethod
    def _encode_image_to_base64(image: bytes | str) -> str:
        """
        Encode image to base64 string if it's bytes.

        :param image: Image as bytes or already base64-encoded string.
        :return: Base64-encoded string.
        """
        if isinstance(image, bytes):
            return base64.b64encode(image).decode("utf-8")
        return image

    @classmethod
    def _validate_image_format(cls, image_format: str) -> str:
        """
        Validate and normalize image format.

        :param image_format: The image format to validate.
        :return: Normalized image format.
        :raises ValueError: If the format is not supported.
        """
        normalized = image_format.lower().strip()
        if normalized == "jpg":
            normalized = "jpeg"
        if normalized not in cls.VALID_IMAGE_FORMATS:
            raise ValueError(
                f"Unsupported image format: {image_format}. "
                f"Supported formats: {cls.VALID_IMAGE_FORMATS}"
            )
        return normalized
