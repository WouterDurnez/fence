"""
Tests for Gemini 2.0 Flash model
"""
import os
import pytest
from fence.models.gemini.gemini import GeminiFlash2_0

# Check if Google Gemini API key is present
has_gemini_api_key = os.environ.get("GOOGLE_GEMINI_API_KEY") is not None

pytestmark = pytest.mark.skip(
    reason="Test ignored because model isn't included in Free Tier. Use a Tiered API Key and enable these tests for verification"
)


@pytest.mark.skipif(
    not has_gemini_api_key, reason="Google Gemini API key not found in environment"
)
def test_gemini_flash_2_0_init():
    """
    Test case for the GeminiFlash2_0 class.
    Checks model_id.
    """
    gemini = GeminiFlash2_0(source="test")
    assert gemini.model_id == "gemini-2.0-flash"


@pytest.mark.skipif(
    not has_gemini_api_key, reason="Google Gemini API key not found in environment"
)
def test_gemini_flash_2_0_integration():
    """
    Integration test for GeminiFlash2_0.
    Calls the API with a simple prompt.
    """
    gemini = GeminiFlash2_0(source="test")
    response = gemini.invoke("Hello, say hi!")
    assert isinstance(response, str)
    assert len(response) > 0
