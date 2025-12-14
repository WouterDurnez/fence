"""
Base class for Embeddings
"""

from abc import ABC, abstractmethod


class Embeddings(ABC):
    """
    Base class for text Embeddings.

    This class provides the interface for text-only embedding models.
    For multimodal embeddings (text + image), see MultimodalEmbeddings.
    """

    model_id: str
    model_name: str
    source: str | None = None

    def __init__(
            self,
            source: str | None = None,
            **kwargs
        ):
        """
        Initialize an Embeddings model
        :param str source: An indicator of where (e.g., which feature) the model is operating from.
        :param **kwargs: Additional keyword arguments
        """

        self.source = source

    def __call__(self, *args, **kwds):
        return self.embed(*args, **kwds)

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Embed text.

        Args:
            text: Text to embed.

        Returns:
            Embedding as a list of floats.
        """


class MultimodalEmbeddings(Embeddings):
    """
    Base class for multimodal embeddings (text + image).

    This class extends Embeddings to support embedding images and
    combined text+image inputs. Models implementing this class can
    embed text, images, or both together in the same semantic space.
    """

    @abstractmethod
    def embed_image(
        self,
        image: bytes | str,
        image_format: str | None = None
    ) -> list[float]:
        """Embed an image.

        Args:
            image: Image data as bytes or base64-encoded string.
            image_format: Optional image format (e.g., 'jpeg', 'png').
                          May be auto-detected if not provided.

        Returns:
            Embedding as a list of floats.
        """

    def embed_multimodal(
        self,
        text: str | None = None,
        image: bytes | str | None = None,
        image_format: str | None = None
    ) -> list[float]:
        """Embed text and/or image together.

        This method embeds text and image inputs into the same semantic space,
        allowing for cross-modal similarity comparisons.

        Note: Not all multimodal embedding models support combined text+image
        embeddings. This method provides a default implementation that raises
        NotImplementedError. Models that support this feature should override it.

        Args:
            text: Optional text to embed.
            image: Optional image data as bytes or base64-encoded string.
            image_format: Optional image format (e.g., 'jpeg', 'png').

        Returns:
            Embedding as a list of floats.

        Raises:
            NotImplementedError: If the model does not support combined embeddings.
            ValueError: If neither text nor image is provided.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support combined text+image embeddings. "
            "Use embed() for text or embed_image() for images separately."
        )
