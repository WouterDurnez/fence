from unittest.mock import MagicMock, patch

import pytest

from fence.embeddings.openai.openai import OpenAIEmbeddingsBase


class MockOpenAIEmbeddings(OpenAIEmbeddingsBase):
    def __init__(self, **kwargs):
        """
        Initialize mock OpenAI embeddings model
        :param kwargs: Additional keyword arguments
        """
        super().__init__(**kwargs)
        self.model_id = "mock_model"

    def _embed(self, text: str) -> dict:
        """
        Override _embed to simulate a successful response without actual OpenAI call
        """
        return {
            "data": [
                {
                    "embedding": [0.1, 0.2, 0.3],
                }
            ]
        }


class TestOpenAIEmbeddingBase:
    @pytest.fixture
    def openai_embedding_base(self):
        """Create a MockOpenAIEmbeddings instance for testing"""
        return MockOpenAIEmbeddings(
            source="test_source",
            full_response=False,
            api_key="test_api_key",
        )

    @pytest.fixture
    def openai_embedding_base_full_response(self):
        """Create a MockOpenAIEmbeddings instance for testing with full response"""
        return MockOpenAIEmbeddings(
            source="test_source",
            full_response=True,
            api_key="test_api_key",
        )

    def test_initialization(self, openai_embedding_base):
        """Test that the OpenAIEmbeddingBase is initialized correctly"""
        assert openai_embedding_base.source == "test_source"
        assert openai_embedding_base.full_response is False
        assert openai_embedding_base.api_key == "test_api_key"
        assert openai_embedding_base.endpoint == "https://api.openai.com/v1"
        assert (
            openai_embedding_base.embedding_endpoint
            == "https://api.openai.com/v1/embeddings"
        )
        assert openai_embedding_base.model_id == "mock_model"

    @patch("requests.post")
    def test_embed(self, mock_post, openai_embedding_base):
        """Test that the OpenAIEmbeddingBase embed method returns the correct embedding"""
        mock_post.return_value = MagicMock()
        mock_post.return_value.json.return_value = {
            "data": [{"embedding": [0.1, 0.2, 0.3]}]
        }

        embedding = openai_embedding_base.embed("test_text")
        assert embedding == [0.1, 0.2, 0.3]

    @patch("requests.post")
    def test_embed_full_response(self, mock_post, openai_embedding_base_full_response):
        """Test that the OpenAIEmbeddingBase embed method returns the correct response"""
        mock_post.return_value = MagicMock()
        mock_post.return_value.json.return_value = {
            "data": [{"embedding": [0.1, 0.2, 0.3]}]
        }

        response = openai_embedding_base_full_response.embed("test_text")
        assert response == {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
