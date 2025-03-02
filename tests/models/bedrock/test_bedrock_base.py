import pytest

from fence.models.base import Messages
from fence.models.bedrock.base import BedrockBase


# Create a mock model class that inherits from BedrockBase
class MockBedrockModel(BedrockBase):
    """Mock Bedrock model for testing"""

    def __init__(self, **kwargs):
        """
        Initialize mock model with a predefined model_id
        :param kwargs: Additional keyword arguments
        """
        super().__init__(**kwargs)
        self.model_id = "MODEL_ID_HAIKU"

    def _invoke(self, prompt: str | Messages, stream=False, **kwargs) -> dict:
        """
        Override _invoke to simulate a successful response without actual AWS call
        """

        self._check_if_prompt_is_valid(prompt)

        return {
            "output": {"message": {"content": [{"text": "Test response"}]}},
            "usage": {"inputTokens": 10, "outputTokens": 20},
        }


class TestBedrockBase:
    @pytest.fixture
    def bedrock_base(self):
        """Create a MockBedrockModel instance for testing"""
        return MockBedrockModel(
            source="test_source",
            full_response=False,
            metric_prefix="test_prefix",
            extra_tags={"test_tag": "value"},
        )

    def test_initialization(self, bedrock_base):
        """Test that the BedrockBase is initialized correctly"""
        assert bedrock_base.source == "test_source"
        assert bedrock_base.metric_prefix == "test_prefix"
        assert bedrock_base.logging_tags == {"test_tag": "value"}
        assert bedrock_base.model_id == "MODEL_ID_HAIKU"

        # Check default model kwargs
        assert bedrock_base.model_kwargs == {
            "temperature": 0.01,
            "maxTokens": 2048,
            "topP": 0.9,
        }

    def test_invoke_with_different_prompt_types(self, bedrock_base):
        """Test invoking with different prompt types"""
        # Test with string prompt
        str_response = bedrock_base.invoke("Test prompt")
        assert str_response == {
            "output": {"message": {"content": [{"text": "Test response"}]}},
            "usage": {"inputTokens": 10, "outputTokens": 20},
        }

        # Test with Messages prompt
        messages = Messages(messages=[{"role": "user", "content": "Test prompt"}])
        msg_response = bedrock_base.invoke(messages)
        assert msg_response == {
            "output": {"message": {"content": [{"text": "Test response"}]}},
            "usage": {"inputTokens": 10, "outputTokens": 20},
        }

    def test_invoke_with_override_kwargs(self, bedrock_base):
        """Test invoking with override kwargs"""
        response = bedrock_base.invoke("Test prompt", temperature=0.7, maxTokens=1024)
        assert response == {
            "output": {"message": {"content": [{"text": "Test response"}]}},
            "usage": {"inputTokens": 10, "outputTokens": 20},
        }

    def test_invalid_prompt_type(self, bedrock_base):
        """Test invoking with an invalid prompt type"""
        with pytest.raises(
            ValueError, match="Prompt must be a string or a Messages object!"
        ):
            print(bedrock_base.invoke(123))  # Invalid type
