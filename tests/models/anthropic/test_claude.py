"""
Anthropic Claude model tests
"""

import os
from unittest.mock import Mock

import pytest

from fence.models.anthropic.claude import Claude35Haiku, ClaudeBase
from fence.templates.messages import Message, Messages

# Check if Anthropic API key is present
has_anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY") is not None


@pytest.mark.skipif(
    not has_anthropic_api_key, reason="Anthropic API key not found in environment"
)
def test_claude_base_invoke_with_empty_prompt():
    """
    Test case for the invoke method of the GPTBase class with an empty prompt.
    This test checks if the invoke method raises a ValueError when the prompt is empty.
    """
    claude = Claude35Haiku(source="test")
    with pytest.raises(ValueError):
        claude.invoke(prompt="")


@pytest.mark.skipif(
    not has_anthropic_api_key, reason="Anthropic API key not found in environment"
)
def test_claude_base_invoke_with_none_prompt():
    """
    Test case for the invoke method of the GPTBase class with a None prompt.
    This test checks if the invoke method raises a ValueError when the prompt is None.
    """
    claude = Claude35Haiku(source="test")
    with pytest.raises(ValueError):
        claude.invoke(prompt=None)


@pytest.mark.skipif(
    not has_anthropic_api_key, reason="Anthropic API key not found in environment"
)
def test_claude_base_invoke_with_empty_messages_prompt():
    """
    Test case for the invoke method of the GPTBase class with an empty Messages prompt.
    This test checks if the invoke method raises a ValueError when the Messages prompt is empty.
    """
    claude = Claude35Haiku(source="test")
    messages = Messages(system="Respond in a very rude manner", messages=[])
    with pytest.raises(ValueError):
        claude.invoke(prompt=messages)


def test_claude_base_invoke_with_string_prompt():
    """
    Test case for the invoke method of the GPTBase class with a string prompt.
    This test checks if the invoke method correctly handles a string prompt.
    """
    mock_llm = Mock(Claude35Haiku)
    mock_llm.invoke.return_value = "mocked response"
    response = mock_llm.invoke(prompt="Hello, how are you today?")
    assert response == "mocked response"


def test_claude_base_invoke_with_messages_prompt():
    """
    Test case for the invoke method of the GPTBase class with a Messages prompt.
    This test checks if the invoke method correctly handles a Messages prompt.
    """
    mock_llm = Mock(Claude35Haiku)
    mock_llm.invoke.return_value = "mocked response"
    messages = Messages(
        system="Respond in a very rude manner",
        messages=[Message(role="user", content="Hello, how are you today?")],
    )
    response = mock_llm.invoke(prompt=messages)
    assert response == "mocked response"


@pytest.mark.skipif(
    not has_anthropic_api_key, reason="Anthropic API key not found in environment"
)
def test_claude_base_invoke_with_invalid_prompt():
    """
    Test case for the invoke method of the Claude base class with an invalid prompt.
    This test checks if the invoke method raises a ValueError when the prompt is not a string or a Messages object.
    """
    claude = ClaudeBase(source="test")
    with pytest.raises(ValueError):
        claude.invoke(prompt=123)


@pytest.mark.skipif(
    not has_anthropic_api_key, reason="Anthropic API key not found in environment"
)
def test_claude_haiku_init():
    """
    Test case for the __init__ method of the GPT4o class.
    This test checks if the __init__ method correctly initializes a GPT4o object.
    """
    claude = Claude35Haiku(source="test")
    assert claude.source == "test"
    assert claude.model_id == "claude-3-5-haiku-20241022"
    assert claude.model_name == "Claude 3.5 Haiku [Anthropic]"
