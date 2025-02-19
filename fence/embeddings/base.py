"""
Base class for Embeddings
"""

from abc import ABC, abstractmethod


class Embeddings(ABC):
    """
    Base class for Embeddings
    """

    model_id: str
    model_name: str
    source: str | None = None,

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
    def embed(self, query: str) -> list[float]:
        """Embed query text.

        Args:
            query: Text to embed.

        Returns:
            Embedding.
        """
