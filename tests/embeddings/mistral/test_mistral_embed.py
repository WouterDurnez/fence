import pytest
from unittest.mock import patch, MagicMock

from fence.embeddings.mistral.mistral import MistralEmbeddingsBase

class MockMistralEmbeddings(MistralEmbeddingsBase):
    def __init__(self, **kwargs):
        """
        Initialize mock Mistral embeddings model
        :param kwargs: Additional keyword arguments
        """
        super().__init__(**kwargs)
        self.model_id = "mistral-embed"

    def _embed(self, text: str) -> dict:
        """
        Override _embed to simulate a successful response without actual Mistral call
        """
        return {
            "data": [
                {
                    "embedding": [0.1, 0.2, 0.3],
                }
            ]
        }

class TestMistralEmbeddingBase:
    @pytest.fixture
    def mistral_embedding_base(self):
        """Create a MockMistralEmbeddings instance for testing"""
        return MockMistralEmbeddings(
            source="test_source",
            full_response=False,
            api_key="test_api_key",
        )
    
    @pytest.fixture
    def mistral_embedding_base_full_response(self):
        """Create a MockMistralEmbeddings instance for testing with full response"""
        return MockMistralEmbeddings(
            source="test_source",
            full_response=True,
            api_key="test_api_key",
        )
    
    def test_initialization(self, mistral_embedding_base):
        """Test that the MistralEmbeddingBase is initialized correctly"""
        assert mistral_embedding_base.source == "test_source"
        assert mistral_embedding_base.full_response is False
        assert mistral_embedding_base.api_key == "test_api_key"
        assert mistral_embedding_base.endpoint == "https://api.mistral.ai/v1"
        assert mistral_embedding_base.embedding_endpoint == "https://api.mistral.ai/v1/embeddings"
        assert mistral_embedding_base.model_id == "mistral-embed"

    @patch("requests.post")
    def test_embed(self, mock_post, mistral_embedding_base):
        """Test that the MistralEmbeddingBase embed method returns the correct embedding"""
        mock_post.return_value = MagicMock()
        mock_post.return_value.json.return_value = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}

        embedding = mistral_embedding_base.embed("test_text")
        assert embedding == [0.1, 0.2, 0.3]

    @patch("requests.post")
    def test_embed_full_response(self, mock_post, mistral_embedding_base_full_response):
        """Test that the MistralEmbeddingBase embed method returns the correct response"""
        mock_post.return_value = MagicMock()
        mock_post.return_value.json.return_value = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}

        response = mistral_embedding_base_full_response.embed("test_text")
        assert response == {"data": [{"embedding": [0.1, 0.2, 0.3]}]} 