"""
BaseTool tests
"""

import logging

import pytest

from fence.tools.base import BaseTool


class DummyTool(BaseTool):
    """A dummy tool for testing purposes."""

    def execute_tool(self, environment: dict = None, **kwargs):
        return f"Executed with environment: {environment}, kwargs: {kwargs}"


def test_base_tool_initialization():
    """Test the initialization of BaseTool."""
    tool = DummyTool(description="Test tool")
    assert tool.description == "Test tool"
    assert tool.environment == {}


def test_base_tool_run():
    """Test the run method of BaseTool."""
    tool = DummyTool()
    result = tool.run(environment={"key": "value"}, arg1="test")
    assert "Executed with environment: {'key': 'value'}" in result
    assert "kwargs: {'arg1': 'test'}" in result


def test_base_tool_abstract_method():
    """Test that BaseTool cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseTool()


def test_format_toml():
    """Test the format_toml method."""
    tool = DummyTool(description="Custom description")
    toml_output = tool.format_toml()
    assert 'tool_name = "DummyTool"' in toml_output
    assert 'tool_description = "Custom description"' in toml_output
    assert "# No arguments" in toml_output


def test_format_toml_no_description():
    """Test the format_toml method when no description is provided."""
    tool = DummyTool()
    toml_output = tool.format_toml()
    assert 'tool_name = "DummyTool"' in toml_output
    assert 'tool_description = "A dummy tool for testing purposes."' in toml_output


def test_format_toml_no_description_no_docstring(caplog):
    """Test the format_toml method when no description and no docstring are provided."""

    class NoDescriptionTool(BaseTool):
        def execute_tool(self, environment: dict = None, **kwargs):
            pass

    tool = NoDescriptionTool()
    with caplog.at_level(logging.WARNING):
        tool.format_toml()
    assert "Tool NoDescriptionTool has no description or docstring." in caplog.text
