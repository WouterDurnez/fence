import pytest
from unittest.mock import patch, MagicMock

from fence.embeddings.google.gemini import GeminiEmbeddingsBase, GeminiEmbeddings


class MockGeminiEmbeddings(GeminiEmbeddingsBase):
    """Mock Gemini embeddings for testing"""

    model_id = "gemini-embedding-001"

    def __init__(self, **kwargs):
        """
        Initialize mock Gemini embeddings model
        :param kwargs: Additional keyword arguments
        """
        super().__init__(**kwargs)

    def _embed(self, text: str) -> dict:
        """
        Override _embed to simulate a successful response without actual Gemini call
        """
        return {
            "embedding": {
                "values": [0.1, 0.2, 0.3],
            }
        }


class TestGeminiEmbeddingsBase:
    @pytest.fixture
    def gemini_embedding_base(self):
        """Create a MockGeminiEmbeddings instance for testing"""
        return MockGeminiEmbeddings(
            source="test_source",
            full_response=False,
            api_key="test_api_key",
        )

    @pytest.fixture
    def gemini_embedding_base_full_response(self):
        """Create a MockGeminiEmbeddings instance for testing with full response"""
        return MockGeminiEmbeddings(
            source="test_source",
            full_response=True,
            api_key="test_api_key",
        )

    @pytest.fixture
    def gemini_embedding_with_options(self):
        """Create a MockGeminiEmbeddings instance with task_type and dimensionality"""
        return MockGeminiEmbeddings(
            source="test_source",
            api_key="test_api_key",
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=768,
        )

    def test_initialization(self, gemini_embedding_base):
        """Test that the GeminiEmbeddingsBase is initialized correctly"""
        assert gemini_embedding_base.source == "test_source"
        assert gemini_embedding_base.full_response is False
        assert gemini_embedding_base.api_key == "test_api_key"
        assert gemini_embedding_base.endpoint == "https://generativelanguage.googleapis.com/v1beta"
        assert gemini_embedding_base.model_id == "gemini-embedding-001"
        assert gemini_embedding_base.task_type is None
        assert gemini_embedding_base.output_dimensionality is None

    def test_initialization_with_options(self, gemini_embedding_with_options):
        """Test initialization with task_type and output_dimensionality"""
        assert gemini_embedding_with_options.task_type == "RETRIEVAL_QUERY"
        assert gemini_embedding_with_options.output_dimensionality == 768

    def test_embed(self, gemini_embedding_base):
        """Test that the GeminiEmbeddingsBase embed method returns the correct embedding"""
        embedding = gemini_embedding_base.embed("test_text")
        assert embedding == [0.1, 0.2, 0.3]

    def test_embed_full_response(self, gemini_embedding_base_full_response):
        """Test that the GeminiEmbeddingsBase embed method returns the correct full response"""
        response = gemini_embedding_base_full_response.embed("test_text")
        assert response == {"embedding": {"values": [0.1, 0.2, 0.3]}}

    def test_api_key_from_env(self):
        """Test that API key is read from environment variable"""
        with patch.dict("os.environ", {"GEMINI_API_KEY": "env_api_key"}):
            embeddings = MockGeminiEmbeddings(source="test")
            assert embeddings.api_key == "env_api_key"

    def test_missing_api_key_raises_error(self):
        """Test that missing API key raises ValueError"""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError) as excinfo:
                MockGeminiEmbeddings(source="test", api_key=None)
            assert "GEMINI_API_KEY" in str(excinfo.value)

    @patch("requests.post")
    def test_embed_api_call(self, mock_post, gemini_embedding_base):
        """Test that the embed method makes the correct API call"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embedding": {"values": [0.4, 0.5, 0.6]}}
        mock_post.return_value = mock_response

        # Use real _embed instead of mocked one
        gemini_embedding_base._embed = GeminiEmbeddingsBase._embed.__get__(
            gemini_embedding_base, GeminiEmbeddingsBase
        )

        embedding = gemini_embedding_base.embed("test_text")

        # Verify the API call
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert "x-goog-api-key" in call_args.kwargs["headers"]
        assert call_args.kwargs["json"]["content"]["parts"][0]["text"] == "test_text"

        assert embedding == [0.4, 0.5, 0.6]

    @patch("requests.post")
    def test_embed_with_task_type(self, mock_post, gemini_embedding_with_options):
        """Test that task_type and output_dimensionality are included in request"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embedding": {"values": [0.1, 0.2, 0.3]}}
        mock_post.return_value = mock_response

        # Use real _embed
        gemini_embedding_with_options._embed = GeminiEmbeddingsBase._embed.__get__(
            gemini_embedding_with_options, GeminiEmbeddingsBase
        )

        gemini_embedding_with_options.embed("test_text")

        call_args = mock_post.call_args
        assert call_args.kwargs["json"]["task_type"] == "RETRIEVAL_QUERY"
        assert call_args.kwargs["json"]["output_dimensionality"] == 768


class TestGeminiEmbeddings:
    def test_model_id(self):
        """Test that GeminiEmbeddings has the correct model_id"""
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test_key"}):
            embeddings = GeminiEmbeddings(source="test")
            assert embeddings.model_id == "gemini-embedding-001"
            assert embeddings.model_name == "Gemini Embedding 001"

