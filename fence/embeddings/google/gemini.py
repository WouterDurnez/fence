"""
Google Gemini Embeddings implementation
"""

import os
from typing import Literal

import requests

from fence.embeddings.base import Embeddings


# Supported task types for Gemini embeddings
TaskType = Literal[
    "SEMANTIC_SIMILARITY",
    "CLASSIFICATION",
    "CLUSTERING",
    "RETRIEVAL_DOCUMENT",
    "RETRIEVAL_QUERY",
    "CODE_RETRIEVAL_QUERY",
    "QUESTION_ANSWERING",
    "FACT_VERIFICATION",
]

# Supported output dimensions
OutputDimensionality = Literal[768, 1536, 3072]


class GeminiEmbeddingsBase(Embeddings):
    """
    Base class for Google Gemini embeddings.
    """

    def __init__(
        self,
        source: str | None = None,
        full_response: bool = False,
        api_key: str | None = None,
        task_type: TaskType | None = None,
        output_dimensionality: OutputDimensionality | None = None,
        **kwargs
    ):
        """
        Initialize a Gemini embeddings model.

        :param str source: An indicator of where (e.g., which feature) the model is operating from.
        :param bool full_response: Whether to return the full response from the Gemini service.
        :param str api_key: Google API key. If not provided, reads from GEMINI_API_KEY env var.
        :param TaskType task_type: The task type for embedding optimization. One of:
            - SEMANTIC_SIMILARITY: Text similarity (recommendations, duplicate detection)
            - CLASSIFICATION: Classify texts (sentiment analysis, spam detection)
            - CLUSTERING: Cluster texts by similarity
            - RETRIEVAL_DOCUMENT: For documents to be retrieved
            - RETRIEVAL_QUERY: For search queries
            - CODE_RETRIEVAL_QUERY: For code search queries
            - QUESTION_ANSWERING: For questions in Q&A systems
            - FACT_VERIFICATION: For fact-checking statements
        :param OutputDimensionality output_dimensionality: Output embedding dimensions.
            Recommended values: 768, 1536, or 3072 (default).
        :param **kwargs: Additional keyword arguments
        """
        super().__init__(source=source)

        self.full_response = full_response
        self.task_type = task_type
        self.output_dimensionality = output_dimensionality

        # Find API key
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", None)
        if not self.api_key:
            raise ValueError(
                "Gemini API key must be provided, either as an argument or in the "
                "environment variable `GEMINI_API_KEY`"
            )

        self.endpoint = "https://generativelanguage.googleapis.com/v1beta"

    def _get_embedding_endpoint(self) -> str:
        """Get the embedding endpoint URL for the model."""
        return f"{self.endpoint}/models/{self.model_id}:embedContent"

    def embed(self, text: str, **kwargs) -> list[float] | dict:
        """
        Embed text using the Gemini embedding model.

        :param str text: Text to embed.
        :param **kwargs: Additional keyword arguments (unused).
        :return: Embedding as a list of floats, or full response dict if full_response=True.
        """
        response = self._embed(text=text)
        embedding = response["embedding"]["values"]

        # Depending on the full_response flag, return either the full response or just the embedding
        return response if self.full_response else embedding

    def _embed(self, text: str) -> dict:
        """
        Internal method to call the Gemini embedding API.

        :param str text: Text to embed.
        :return: Response dictionary from the API.
        """
        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json"
        }

        data = {
            "model": f"models/{self.model_id}",
            "content": {"parts": [{"text": text}]}
        }

        # Add optional parameters
        if self.task_type:
            data["task_type"] = self.task_type
        if self.output_dimensionality:
            data["output_dimensionality"] = self.output_dimensionality

        try:
            response = requests.post(
                self._get_embedding_endpoint(),
                headers=headers,
                json=data
            )

            # Check status code
            if response.status_code != 200:
                raise ValueError(f"Error raised by Gemini service: {response.text}")

            return response.json()

        except requests.RequestException as e:
            raise ValueError(f"Error raised by Gemini service: {e}")


class GeminiEmbeddings(GeminiEmbeddingsBase):
    """
    Google Gemini Text Embedding model.

    Uses the gemini-embedding-001 model which generates 3072-dimensional embeddings
    by default, with optional truncation to 768 or 1536 dimensions.

    Example:
        >>> gemini = GeminiEmbeddings(source="search")
        >>> embedding = gemini.embed("What is the meaning of life?")
        >>> print(f"Embedding dimension: {len(embedding)}")

        >>> # With task-specific optimization
        >>> gemini = GeminiEmbeddings(
        ...     source="search",
        ...     task_type="RETRIEVAL_QUERY",
        ...     output_dimensionality=768
        ... )
        >>> query_embedding = gemini.embed("How does photosynthesis work?")
    """

    model_id = "gemini-embedding-001"
    model_name = "Gemini Embedding 001"

    def __init__(self, source: str | None = None, **kwargs):
        """
        Initialize a Gemini Text Embedding model.

        :param str source: An indicator of where (e.g., which feature) the model is operating from.
        :param **kwargs: Additional keyword arguments passed to GeminiEmbeddingsBase.
        """
        super().__init__(source=source, **kwargs)

