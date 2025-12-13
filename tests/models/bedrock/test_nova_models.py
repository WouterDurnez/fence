"""
Tests for Amazon Nova models with context window support.

This module tests:
1. Backward compatibility (default context windows)
2. Context window parameter validation
3. Model ID and name generation with context windows
4. Invalid context window rejection
"""

import pytest

from fence.models.bedrock.nova import (
    NovaPro,
    NovaLite,
    NovaMicro,
    Nova2Lite,
    Nova2Lite256K,
)


class TestNovaProContextWindows:
    """Test NovaPro model with context window options."""

    def test_default_context_window(self):
        """Test NovaPro with default context window (no suffix)."""
        model = NovaPro(source="test")
        assert model.model_id == "amazon.nova-pro-v1:0"
        assert model.model_name == "Nova Pro"

    def test_context_window_24k(self):
        """Test NovaPro with 24k context window."""
        model = NovaPro(context_window="24k", source="test")
        assert model.model_id == "amazon.nova-pro-v1:0:24k"
        assert model.model_name == "Nova Pro (24k)"

    def test_context_window_300k(self):
        """Test NovaPro with 300k context window."""
        model = NovaPro(context_window="300k", source="test")
        assert model.model_id == "amazon.nova-pro-v1:0:300k"
        assert model.model_name == "Nova Pro (300k)"

    def test_invalid_context_window(self):
        """Test NovaPro rejects invalid context window."""
        with pytest.raises(ValueError) as exc_info:
            NovaPro(context_window="invalid", source="test")
        assert "Invalid context_window 'invalid'" in str(exc_info.value)
        assert "Valid options: ['24k', '300k']" in str(exc_info.value)


class TestNovaLiteContextWindows:
    """Test NovaLite model with context window options."""

    def test_default_context_window(self):
        """Test NovaLite with default context window."""
        model = NovaLite(source="test")
        assert model.model_id == "amazon.nova-lite-v1:0"
        assert model.model_name == "Nova Lite"

    def test_context_window_24k(self):
        """Test NovaLite with 24k context window."""
        model = NovaLite(context_window="24k", source="test")
        assert model.model_id == "amazon.nova-lite-v1:0:24k"
        assert model.model_name == "Nova Lite (24k)"

    def test_context_window_300k(self):
        """Test NovaLite with 300k context window."""
        model = NovaLite(context_window="300k", source="test")
        assert model.model_id == "amazon.nova-lite-v1:0:300k"
        assert model.model_name == "Nova Lite (300k)"

    def test_invalid_context_window(self):
        """Test NovaLite rejects invalid context window."""
        with pytest.raises(ValueError) as exc_info:
            NovaLite(context_window="128k", source="test")
        assert "Invalid context_window '128k'" in str(exc_info.value)


class TestNovaMicroContextWindows:
    """Test NovaMicro model with context window options."""

    def test_default_context_window(self):
        """Test NovaMicro with default context window."""
        model = NovaMicro(source="test")
        assert model.model_id == "amazon.nova-micro-v1:0"
        assert model.model_name == "Nova Micro"

    def test_context_window_24k(self):
        """Test NovaMicro with 24k context window."""
        model = NovaMicro(context_window="24k", source="test")
        assert model.model_id == "amazon.nova-micro-v1:0:24k"
        assert model.model_name == "Nova Micro (24k)"

    def test_context_window_128k(self):
        """Test NovaMicro with 128k context window."""
        model = NovaMicro(context_window="128k", source="test")
        assert model.model_id == "amazon.nova-micro-v1:0:128k"
        assert model.model_name == "Nova Micro (128k)"

    def test_invalid_context_window(self):
        """Test NovaMicro rejects invalid context window."""
        with pytest.raises(ValueError) as exc_info:
            NovaMicro(context_window="300k", source="test")
        assert "Invalid context_window '300k'" in str(exc_info.value)


class TestNova2LiteContextWindows:
    """Test Nova2Lite model with context window options."""

    def test_default_context_window(self):
        """Test Nova2Lite with default context window."""
        model = Nova2Lite(source="test")
        assert model.model_id == "amazon.nova-2-lite-v1:0"
        assert model.model_name == "Nova 2 Lite"

    def test_context_window_256k(self):
        """Test Nova2Lite with 256k context window."""
        model = Nova2Lite(context_window="256k", source="test")
        assert model.model_id == "amazon.nova-2-lite-v1:0:256k"
        assert model.model_name == "Nova 2 Lite (256k)"

    def test_invalid_context_window(self):
        """Test Nova2Lite rejects invalid context window."""
        with pytest.raises(ValueError) as exc_info:
            Nova2Lite(context_window="24k", source="test")
        assert "Invalid context_window '24k'" in str(exc_info.value)


class TestNova2Lite256KBackwardCompatibility:
    """Test Nova2Lite256K for backward compatibility."""

    def test_nova2_lite_256k_deprecated_class(self):
        """Test Nova2Lite256K class still works for backward compatibility."""
        model = Nova2Lite256K(source="test")
        assert model.model_id == "amazon.nova-2-lite-v1:0:256k"
        assert model.model_name == "Nova 2 Lite (256k)"

