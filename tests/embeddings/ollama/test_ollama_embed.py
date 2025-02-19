import shutil

import pytest

from fence.embeddings.ollama.ollama import OllamaEmbeddingBase

ollama_installed = shutil.which("ollama") is not None


class MockOllamaEmbeddings(OllamaEmbeddingBase):
    def __init__(self, **kwargs):
        """
        Initialize mock Ollama embeddings model
        :param kwargs: Additional keyword arguments
        """
        super().__init__(**kwargs)
        self.model_id = "embedder"

    def _embed(self, text: str) -> dict:
        """
        Override _embed to simulate a successful response without actual Ollama call
        """
        return {
            "embedding": [0.1, 0.2, 0.3],
        }


class TestOllamaEmbeddingBase:
    @pytest.fixture
    def ollama_embedding_base(self):
        """Create a MockOllamaEmbeddings instance for testing"""
        return MockOllamaEmbeddings(
            source="test_source",
            full_response=False,
            endpoint="http://localhost:11434/api",
        )

    @pytest.fixture
    def ollama_embedding_base_full_response(self):
        """Create a MockOllamaEmbeddings instance for testing with full response"""
        return MockOllamaEmbeddings(
            source="test_source",
            full_response=True,
            endpoint="http://localhost:11434/api",
        )

    @pytest.mark.skipif(
        not ollama_installed, reason="Ollama is not installed in the CI environment"
    )
    def test_initialization(self, ollama_embedding_base):
        """Test that the OllamaEmbeddingBase is initialized correctly"""
        assert ollama_embedding_base.source == "test_source"
        assert ollama_embedding_base.full_response is False
        assert ollama_embedding_base.endpoint == "http://localhost:11434/api"
        assert (
            ollama_embedding_base.embedding_endpoint
            == "http://localhost:11434/api/embeddings"
        )
        assert ollama_embedding_base.pull_endpoint == "http://localhost:11434/api/pull"
        assert ollama_embedding_base.model_id == "embedder"

    @pytest.mark.skipif(
        not ollama_installed, reason="Ollama is not installed in the CI environment"
    )
    def test_embed(self, ollama_embedding_base):
        """Test the embed method of the OllamaEmbeddingBase class"""
        embedding = ollama_embedding_base.embed("test text")
        assert embedding == [0.1, 0.2, 0.3]

    @pytest.mark.skipif(
        not ollama_installed, reason="Ollama is not installed in the CI environment"
    )
    def test_embed_full_response(self, ollama_embedding_base_full_response):
        """Test the embed method of the OllamaEmbeddingBase class with full response"""
        response = ollama_embedding_base_full_response.embed("test text")
        assert response == {"embedding": [0.1, 0.2, 0.3]}
