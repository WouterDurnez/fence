"""
Nova Multimodal Embeddings for AWS Bedrock.

This module provides the Nova Multimodal Embeddings model which supports:
- Text embeddings
- Image embeddings
- Video embeddings (async)

Note: Nova generates embeddings for text and images in the same semantic space,
but does not support combined text+image embeddings in a single request.
"""

import logging
from typing import Literal

from fence.embeddings.bedrock.base import BedrockMultimodalEmbeddingsBase

logger = logging.getLogger(__name__)

# Model ID
MODEL_ID_AMAZON_NOVA_MULTIMODAL_EMBEDDINGS_V1 = "amazon.nova-2-multimodal-embeddings-v1:0"

# Valid embedding dimensions for Nova
NOVA_EMBEDDING_DIMENSIONS = (256, 512, 1024, 3072)

# Embedding purposes
EmbeddingPurpose = Literal["GENERIC_INDEX", "GENERIC_QUERY"]

# Text truncation modes
TruncationMode = Literal["START", "END"]


class NovaMultimodalEmbeddings(BedrockMultimodalEmbeddingsBase):
    """
    Amazon Nova Multimodal Embeddings model.

    This model generates embeddings for text, images, and video in the same
    semantic space. Supports both synchronous (text/image) and asynchronous
    (video) embedding generation.

    Features:
    - Text embeddings with configurable truncation
    - Image embeddings (JPEG, PNG, GIF, WebP)
    - Video embeddings via async API (MP4, MOV, MKV, WebM, etc.)
    - Configurable embedding dimensions (256, 512, 1024, 3072)

    Example:
        >>> nova = NovaMultimodalEmbeddings(source="search", embedding_dimension=1024)
        >>> text_emb = nova.embed("A red sports car")
        >>> image_emb = nova.embed_image(image_bytes, image_format="jpeg")
    """

    # Valid video formats for Nova
    VALID_VIDEO_FORMATS = {"mp4", "mov", "mkv", "webm", "flv", "mpeg", "mpg", "wmv", "3gp"}

    def __init__(
        self,
        source: str | None = None,
        embedding_dimension: Literal[256, 512, 1024, 3072] = 1024,
        embedding_purpose: EmbeddingPurpose = "GENERIC_INDEX",
        truncation_mode: TruncationMode = "END",
        **kwargs
    ):
        """
        Initialize Nova Multimodal Embeddings model.

        :param str source: An indicator of where the model is operating from.
        :param int embedding_dimension: Output embedding dimensions (256, 512, 1024, or 3072).
        :param str embedding_purpose: Purpose of embeddings ("GENERIC_INDEX" or "GENERIC_QUERY").
        :param str truncation_mode: How to truncate long text ("START" or "END").
        :param **kwargs: Additional arguments passed to BedrockMultimodalEmbeddingsBase.
        """
        super().__init__(source=source, **kwargs)

        if embedding_dimension not in NOVA_EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"Invalid embedding_dimension: {embedding_dimension}. "
                f"Must be one of {NOVA_EMBEDDING_DIMENSIONS}"
            )

        self.model_id = MODEL_ID_AMAZON_NOVA_MULTIMODAL_EMBEDDINGS_V1
        self.model_name = "Nova Multimodal Embeddings"
        self.embedding_dimension = embedding_dimension
        self.embedding_purpose = embedding_purpose
        self.truncation_mode = truncation_mode

    def _build_base_params(self) -> dict:
        """Build the base embedding parameters."""
        return {
            "embeddingPurpose": self.embedding_purpose,
            "embeddingDimension": self.embedding_dimension,
        }

    def embed(self, text: str) -> list[float]:
        """
        Embed text.

        :param text: Text to embed.
        :return: Embedding as a list of floats.
        """
        self._check_if_text_is_valid(text)

        params = self._build_base_params()
        params["text"] = {
            "truncationMode": self.truncation_mode,
            "value": text,
        }

        request_body = {
            "taskType": "SINGLE_EMBEDDING",
            "singleEmbeddingParams": params,
        }

        response = self._invoke_model(request_body)
        # Nova returns embeddings in {"embeddings": [{"embeddingType": "...", "embedding": [...]}]}
        embedding = response["embeddings"][0]["embedding"]

        return response if self.full_response else embedding

    def embed_image(
        self,
        image: bytes | str,
        image_format: str = "jpeg"
    ) -> list[float]:
        """
        Embed an image.

        :param image: Image as bytes or base64-encoded string.
        :param image_format: Image format ('jpeg', 'png', 'gif', 'webp').
        :return: Embedding as a list of floats.
        """
        image_format = self._validate_image_format(image_format)
        image_base64 = self._encode_image_to_base64(image)

        params = self._build_base_params()
        params["image"] = {
            "format": image_format,
            "source": {"bytes": image_base64},
        }

        request_body = {
            "taskType": "SINGLE_EMBEDDING",
            "singleEmbeddingParams": params,
        }

        response = self._invoke_model(request_body)
        # Nova returns embeddings in {"embeddings": [{"embeddingType": "...", "embedding": [...]}]}
        embedding = response["embeddings"][0]["embedding"]

        return response if self.full_response else embedding

    def start_video_embedding_job(
        self,
        video_s3_uri: str,
        output_s3_uri: str,
        video_format: str = "mp4",
        segment_duration_seconds: int = 15,
        embedding_mode: Literal["VIDEO_ONLY", "AUDIO_ONLY", "AUDIO_VIDEO_COMBINED"] = "AUDIO_VIDEO_COMBINED",
    ) -> str:
        """
        Start an async video embedding job.

        Video embeddings are generated asynchronously due to processing time.
        The embeddings are saved to the specified S3 output location.

        :param video_s3_uri: S3 URI of the video (e.g., 's3://bucket/video.mp4').
        :param output_s3_uri: S3 URI for output embeddings.
        :param video_format: Video format ('mp4', 'mov', 'mkv', 'webm', etc.).
        :param segment_duration_seconds: Duration of video segments in seconds.
        :param embedding_mode: What to embed - video, audio, or both combined.
        :return: The invocation ARN for tracking the async job.

        Example:
            >>> nova = NovaMultimodalEmbeddings(source="video-search")
            >>> job_arn = nova.start_video_embedding_job(
            ...     video_s3_uri="s3://my-bucket/video.mp4",
            ...     output_s3_uri="s3://my-bucket/embeddings/",
            ...     segment_duration_seconds=15
            ... )
            >>> # Check job status later with get_video_embedding_job_status()
        """
        video_format = video_format.lower()
        if video_format not in self.VALID_VIDEO_FORMATS:
            raise ValueError(
                f"Unsupported video format: {video_format}. "
                f"Supported formats: {self.VALID_VIDEO_FORMATS}"
            )

        model_input = {
            "taskType": "SEGMENTED_EMBEDDING",
            "segmentedEmbeddingParams": {
                "embeddingPurpose": self.embedding_purpose,
                "embeddingDimension": self.embedding_dimension,
                "video": {
                    "format": video_format,
                    "embeddingMode": embedding_mode,
                    "source": {
                        "s3Location": {"uri": video_s3_uri}
                    },
                    "segmentationConfig": {
                        "durationSeconds": segment_duration_seconds
                    },
                },
            },
        }

        try:
            response = self.client.start_async_invoke(
                modelId=self.model_id,
                modelInput=model_input,
                outputDataConfig={
                    "s3OutputDataConfig": {
                        "s3Uri": output_s3_uri
                    }
                },
            )
            return response["invocationArn"]
        except Exception as e:
            raise ValueError(f"Error starting video embedding job: {e}")

    def get_video_embedding_job_status(self, invocation_arn: str) -> dict:
        """
        Get the status of an async video embedding job.

        :param invocation_arn: The invocation ARN returned by start_video_embedding_job.
        :return: Dictionary with job status and details.
        """
        try:
            response = self.client.get_async_invoke(invocationArn=invocation_arn)
            return {
                "status": response.get("status"),
                "failure_message": response.get("failureMessage"),
                "submit_time": response.get("submitTime"),
                "end_time": response.get("endTime"),
                "output_location": response.get("outputDataConfig", {}).get("s3OutputDataConfig", {}).get("s3Uri"),
            }
        except Exception as e:
            raise ValueError(f"Error getting video embedding job status: {e}")


if __name__ == "__main__":
    # Example usage - Text embedding
    nova = NovaMultimodalEmbeddings(
        source="example",
        embedding_dimension=1024,
        region="us-east-1"
    )

    text_embedding = nova.embed("Amazon Nova is a multimodal foundation model")
    print(f"Text embedding dimension: {len(text_embedding)}")

    # Example - Image embedding would require actual image bytes
    # image_embedding = nova.embed_image(image_bytes, image_format="jpeg")
