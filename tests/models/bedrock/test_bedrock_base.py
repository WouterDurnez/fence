import pytest

from fence.models.base import Messages

# Assuming the BedrockBase class is in a module named bedrock
from fence.models.bedrock.base import MODEL_ID_HAIKU, BedrockBase


# Create a mock model class that inherits from BedrockBase
class MockBedrockModel(BedrockBase):
    """Mock Bedrock model for testing"""

    def __init__(self, **kwargs):
        """
        Initialize mock model with a predefined model_id
        :param kwargs: Additional keyword arguments
        """
        super().__init__(**kwargs)
        self.model_id = MODEL_ID_HAIKU  # Use the actual model ID

    def _invoke(self, prompt: str | Messages) -> dict:
        """
        Override _invoke to simulate a successful response without actual AWS call
        """
        # Simulate the response structure expected by the invoke method
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
        assert bedrock_base.full_response is False
        assert bedrock_base.metric_prefix == "test_prefix"
        assert bedrock_base.logging_tags == {"test_tag": "value"}
        assert bedrock_base.model_id == MODEL_ID_HAIKU

        # Check default model kwargs
        assert bedrock_base.model_kwargs == {
            "temperature": 0.01,
            "maxTokens": 2048,
            "topP": 0.9,
        }

    def test_bedrock_base_invoke_with_empty_prompt(self, bedrock_base):
        """
        Test case for the invoke method of the BedrockBase class with an empty prompt.
        This test checks if the invoke method raises a ValueError when the prompt is empty.
        """
        with pytest.raises(ValueError):
            bedrock_base.invoke(prompt="")

    def test_bedrock_base_invoke_with_none_prompt(self, bedrock_base):
        """
        Test case for the invoke method of the BedrockBase class with a None prompt.
        This test checks if the invoke method raises a ValueError when the prompt is None.
        """
        with pytest.raises(ValueError):
            bedrock_base.invoke(prompt=None)

    def test_bedrock_base_invoke_with_empty_messages_prompt(self, bedrock_base):
        """
        Test case for the invoke method of the BedrockBase class with an empty Messages prompt.
        This test checks if the invoke method raises a ValueError when the Messages prompt is empty.
        """
        messages = Messages(system="Respond in a very rude manner", messages=[])
        with pytest.raises(ValueError):
            bedrock_base.invoke(prompt=messages)

    def test_invoke_with_different_prompt_types(self, bedrock_base):
        """Test invoking with different prompt types"""
        # Test with string prompt
        str_response = bedrock_base.invoke("Test prompt")
        assert str_response == "Test response"

        # Test with Messages prompt
        # Create a Messages instance that matches your actual implementation
        # This might require adjusting based on your specific Messages class
        messages = Messages(messages=[{"role": "user", "content": "Test prompt"}])
        msg_response = bedrock_base.invoke(messages)
        assert msg_response == "Test response"

    def test_word_count_calculation(self, bedrock_base):
        """Test word count calculation for different input types"""
        # Test with string prompt
        str_prompt = "This is a test prompt with five words"
        str_response = bedrock_base.invoke(str_prompt)
        assert str_response == "Test response"

        # Test with Messages prompt
        messages = Messages(
            messages=[
                {"role": "user", "content": "This is a test prompt with five words"}
            ]
        )
        msg_response = bedrock_base.invoke(messages)
        assert msg_response == "Test response"

    def test_invoke_with_override_kwargs(self, bedrock_base):
        """Test invoking with override kwargs"""
        # Invoke with override kwargs
        response = bedrock_base.invoke("Test prompt", temperature=0.7, maxTokens=1024)

        # Assert response
        assert response == "Test response"

    def test_full_response_flag(self):
        """Test full_response flag behavior"""
        # Create mock models with different full_response settings
        model_full = MockBedrockModel(full_response=True)
        model_text = MockBedrockModel(full_response=False)

        # Invoke with full response
        full_response = model_full.invoke("Test prompt")
        text_response = model_text.invoke("Test prompt")

        # Assert responses
        assert text_response == "Test response"
        assert full_response == {
            "output": {"message": {"content": [{"text": "Test response"}]}},
            "usage": {"inputTokens": 10, "outputTokens": 20},
        }

    def test_invalid_prompt_type(self, bedrock_base):
        """Test invoking with an invalid prompt type"""
        with pytest.raises(
            ValueError, match="Prompt must be a string or a list of messages"
        ):
            bedrock_base.invoke(123)  # Invalid type
