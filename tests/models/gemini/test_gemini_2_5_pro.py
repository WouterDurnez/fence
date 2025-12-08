"""
Tests for Gemini 2.5 Pro model
"""
import os
import pytest
from fence.models.gemini.gemini import Gemini2_5_Pro

# Check if Google Gemini API key is present
has_gemini_api_key = os.environ.get("GOOGLE_GEMINI_API_KEY") is not None

pytestmark = pytest.mark.skip(
    reason="Test ignored because model isn't included in Free Tier. Use a Tiered API Key and enable these tests for verification"
)


@pytest.mark.skipif(
    not has_gemini_api_key, reason="Google Gemini API key not found in environment"
)
def test_gemini_2_5_pro_init():
    """
    Test case for the Gemini2_5_Pro class.
    Checks model_id, model_name, and source.
    """
    gemini = Gemini2_5_Pro(source="test")
    assert gemini.source == "test"
    assert gemini.model_id == "gemini-2.5-pro"
    assert gemini.model_name == "Gemini 2.5 Pro"


@pytest.mark.skipif(
    not has_gemini_api_key, reason="Google Gemini API key not found in environment"
)
def test_gemini_2_5_pro_integration():
    """
    Integration test for Gemini2_5_Pro.
    Calls the API with a simple prompt.
    """
    gemini = Gemini2_5_Pro(source="test")
    response = gemini.invoke("Hello, say hi!")
    assert isinstance(response, str)
    assert len(response) > 0
