"""
Tests for Gemini 2.5 Flash Lite model
"""
import os
import pytest
from fence.models.gemini.gemini import Gemini2_5_FlashLite

# Check if Google Gemini API key is present
has_gemini_api_key = os.environ.get("GOOGLE_GEMINI_API_KEY") is not None


@pytest.mark.skipif(
    not has_gemini_api_key, reason="Google Gemini API key not found in environment"
)
def test_gemini_2_5_flash_lite_init():
    """
    Test case for the Gemini2_5_FlashLite class.
    Checks model_id, model_name, and source.
    """
    gemini = Gemini2_5_FlashLite(source="test")
    assert gemini.source == "test"
    assert gemini.model_id == "gemini-2.5-flash-lite"
    assert gemini.model_name == "Gemini 2.5 Flash Lite"


@pytest.mark.skipif(
    not has_gemini_api_key, reason="Google Gemini API key not found in environment"
)
def test_gemini_2_5_flash_lite_integration():
    """
    Integration test for Gemini2_5_FlashLite.
    Calls the API with a simple prompt.
    """
    gemini = Gemini2_5_FlashLite(source="test")
    response = gemini.invoke("Hello, say hi!")
    assert isinstance(response, str)
    assert len(response) > 0
