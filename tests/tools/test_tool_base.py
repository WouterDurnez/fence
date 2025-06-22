"""
BaseTool tests
"""

import logging

import pytest

from fence.tools.base import BaseTool, tool


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


# Tool decorator tests


def test_tool_decorator_with_description():
    """Test @tool decorator with explicit description."""

    @tool("Custom tool description")
    def sample_function(arg1: str, arg2: int = 10):
        """Original docstring."""
        return f"arg1: {arg1}, arg2: {arg2}"

    # Test tool properties
    assert sample_function.get_tool_name() == "SampleFunctionTool"
    assert sample_function.get_tool_description() == "Custom tool description"

    # Test execution
    result = sample_function.run(arg1="test", arg2=20)
    assert result == "arg1: test, arg2: 20"

    # Test with default parameter
    result = sample_function.run(arg1="test")
    assert result == "arg1: test, arg2: 10"


def test_tool_decorator_with_empty_parentheses():
    """Test @tool() decorator using function docstring."""

    @tool()
    def another_function(city: str):
        """Get information about a city."""
        return f"Information about {city}"

    # Test tool properties
    assert another_function.get_tool_name() == "AnotherFunctionTool"
    assert another_function.get_tool_description() == "Get information about a city."

    # Test execution
    result = another_function.run(city="Paris")
    assert result == "Information about Paris"


def test_tool_decorator_without_parentheses():
    """Test @tool decorator without parentheses using function docstring."""

    @tool
    def calculate_area(width: float, height: float):
        """Calculate the area of a rectangle."""
        return width * height

    # Test tool properties
    assert calculate_area.get_tool_name() == "CalculateAreaTool"
    assert calculate_area.get_tool_description() == "Calculate the area of a rectangle."

    # Test execution
    result = calculate_area.run(width=5.0, height=3.0)
    assert result == 15.0


def test_tool_decorator_no_docstring():
    """Test @tool decorator when function has no docstring."""

    @tool
    def no_docstring_function(value: str):
        return f"Processed: {value}"

    # Should fall back to default description when no docstring
    assert no_docstring_function.get_tool_name() == "NoDocstringFunctionTool"
    assert no_docstring_function.get_tool_description() == "No description provided"

    # Test execution still works
    result = no_docstring_function.run(value="test")
    assert result == "Processed: test"


def test_tool_decorator_parameter_filtering():
    """Test that decorated tool properly filters parameters."""

    @tool
    def specific_params(name: str, age: int):
        """Function with specific parameters."""
        return f"{name} is {age} years old"

    # Should work with correct parameters
    result = specific_params.run(name="Alice", age=30)
    assert result == "Alice is 30 years old"

    # Should filter out extra parameters (like environment)
    result = specific_params.run(
        name="Bob", age=25, environment={"test": "value"}, extra_param="ignored"
    )
    assert result == "Bob is 25 years old"


def test_tool_decorator_complex_function_name():
    """Test tool name generation from complex function names."""

    @tool
    def get_user_profile_data(user_id: int):
        """Get user profile data."""
        return f"Profile for user {user_id}"

    assert get_user_profile_data.get_tool_name() == "GetUserProfileDataTool"


def test_tool_decorator_already_ends_with_tool():
    """Test that function names ending with 'tool' don't get double suffix."""

    @tool
    def validation_tool(data: str):
        """Validate some data."""
        return f"Validated: {data}"

    assert validation_tool.get_tool_name() == "ValidationTool"


def test_tool_decorator_get_tool_params():
    """Test that decorated tool returns correct parameters."""

    @tool
    def multi_param_function(
        required_param: str, optional_param: int = 42, another_optional: bool = True
    ):
        """Function with multiple parameters."""
        return f"required: {required_param}, optional: {optional_param}, bool: {another_optional}"

    params = multi_param_function.get_tool_params()

    # Check parameter names
    param_names = list(params.keys())
    assert "required_param" in param_names
    assert "optional_param" in param_names
    assert "another_optional" in param_names

    # Check that environment and kwargs are not in the function's params
    assert "environment" not in param_names
    assert "kwargs" not in param_names


def test_tool_decorator_bedrock_converse_format():
    """Test that decorated tool can be formatted for Bedrock Converse."""

    @tool("A tool for testing bedrock format")
    def bedrock_test_function(message: str, count: int = 1):
        """Test function for bedrock."""
        return f"Message: {message}, Count: {count}"

    bedrock_format = bedrock_test_function.model_dump_bedrock_converse()

    # Check structure
    assert "toolSpec" in bedrock_format
    tool_spec = bedrock_format["toolSpec"]

    assert tool_spec["name"] == "BedrockTestFunctionTool"
    assert tool_spec["description"] == "A tool for testing bedrock format"

    # Check input schema
    assert "inputSchema" in tool_spec
    assert "json" in tool_spec["inputSchema"]
    json_schema = tool_spec["inputSchema"]["json"]

    assert json_schema["type"] == "object"
    assert "properties" in json_schema
    assert "required" in json_schema

    # Check properties
    properties = json_schema["properties"]
    assert "message" in properties
    assert "count" in properties
    assert properties["message"]["type"] == "string"
    assert properties["count"]["type"] == "integer"

    # Check required fields
    assert "message" in json_schema["required"]
    assert "count" not in json_schema["required"]  # Has default value


def test_tool_decorator_invalid_usage():
    """Test that invalid decorator usage raises appropriate errors."""

    with pytest.raises(ValueError, match="Invalid usage of @tool decorator"):

        @tool(123)  # Invalid type
        def invalid_function():
            pass
