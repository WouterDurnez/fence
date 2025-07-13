"""
BaseTool tests
"""

import logging

import pytest

from fence.tools.base import BaseTool, ToolParameter, tool


class DummyTool(BaseTool):
    """A dummy tool for testing purposes."""

    def execute_tool(self, environment: dict = None, **kwargs):
        return f"Executed with environment: {environment}, kwargs: {kwargs}"


def test_base_tool_initialization():
    """Test the initialization of BaseTool."""
    tool = DummyTool(description="Test tool")
    assert tool.description == "Test tool"
    assert tool.environment == {}
    assert isinstance(tool.parameters, dict)


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


# Tests for unified parameters approach


def test_unified_parameters_basetool():
    """Test unified parameters extraction from BaseTool subclass."""

    class DocumentedTool(BaseTool):
        """A tool with documented parameters."""

        def execute_tool(
            self,
            name: str,
            age: int,
            active: bool = True,
            environment: dict = None,
            **kwargs,
        ):
            """Execute the tool with documented parameters.

            :param name: The person's name as a string
            :param age: The person's age in years
            :param active: Whether the person is currently active
                This parameter controls the active state
            """
            return f"Hello {name}, age {age}, active: {active}"

    tool = DocumentedTool()

    # Test that parameters are ToolParameter objects
    assert isinstance(tool.parameters, dict)
    assert "name" in tool.parameters
    assert "age" in tool.parameters
    assert "active" in tool.parameters

    # Test ToolParameter properties
    name_param = tool.parameters["name"]
    assert isinstance(name_param, ToolParameter)
    assert name_param.name == "name"
    assert name_param.type_annotation is str
    assert name_param.description == "The person's name as a string"
    assert name_param.required

    age_param = tool.parameters["age"]
    assert age_param.name == "age"
    assert age_param.type_annotation is int
    assert age_param.description == "The person's age in years"
    assert age_param.required

    active_param = tool.parameters["active"]
    assert active_param.name == "active"
    assert active_param.type_annotation is bool
    assert (
        active_param.description
        == "Whether the person is currently active This parameter controls the active state"
    )
    assert not active_param.required
    assert active_param.default_value


def test_unified_parameters_decorated():
    """Test unified parameters extraction from decorated tool."""

    @tool("A documented tool")
    def documented_function(city: str, temperature: float, humidity: int = 50):
        """Get weather information for a city.

        :param city: The name of the city to get weather for
        :param temperature: The current temperature in degrees
        :param humidity: The humidity percentage
        """
        return f"Weather in {city}: {temperature}°C, {humidity}% humidity"

    # Test that parameters are ToolParameter objects
    assert isinstance(documented_function.parameters, dict)
    assert "city" in documented_function.parameters
    assert "temperature" in documented_function.parameters
    assert "humidity" in documented_function.parameters

    # Test ToolParameter properties
    city_param = documented_function.parameters["city"]
    assert isinstance(city_param, ToolParameter)
    assert city_param.name == "city"
    assert city_param.type_annotation is str
    assert city_param.description == "The name of the city to get weather for"
    assert city_param.required

    temp_param = documented_function.parameters["temperature"]
    assert temp_param.name == "temperature"
    assert temp_param.type_annotation is float
    assert temp_param.description == "The current temperature in degrees"
    assert temp_param.required

    humidity_param = documented_function.parameters["humidity"]
    assert humidity_param.name == "humidity"
    assert humidity_param.type_annotation is int
    assert humidity_param.description == "The humidity percentage"
    assert not humidity_param.required
    assert humidity_param.default_value == 50


def test_unified_parameters_no_docstring():
    """Test unified parameters extraction when no docstring exists."""

    class UndocumentedTool(BaseTool):
        """A tool without parameter documentation."""

        def execute_tool(self, name: str, age: int, environment: dict = None, **kwargs):
            return f"Hello {name}, age {age}"

    tool = UndocumentedTool()

    # Test that parameters are ToolParameter objects
    assert isinstance(tool.parameters, dict)
    assert "name" in tool.parameters
    assert "age" in tool.parameters

    # Test ToolParameter properties (no descriptions)
    name_param = tool.parameters["name"]
    assert isinstance(name_param, ToolParameter)
    assert name_param.name == "name"
    assert name_param.type_annotation is str
    assert name_param.description is None
    assert name_param.required


def test_explicit_parameters():
    """Test that explicit ToolParameter objects are used."""

    class ExplicitTool(BaseTool):
        """A tool with explicit parameters."""

        def execute_tool(
            self,
            name: str,
            age: int,
            active: bool = True,
            environment: dict = None,
            **kwargs,
        ):
            """Execute the tool.

            :param name: This docstring description should be ignored
            :param age: This docstring description should also be ignored
            """
            return f"Hello {name}"

    # Create tool with explicit ToolParameter objects
    explicit_parameters = {
        "name": ToolParameter(
            name="name",
            type_annotation=str,
            description="Explicit name description",
            required=True,
        ),
        "age": ToolParameter(
            name="age",
            type_annotation=int,
            description="Explicit age description",
            required=True,
        ),
        "active": ToolParameter(
            name="active",
            type_annotation=bool,
            description="Explicit active description",
            required=False,
            default_value=True,
        ),
    }

    tool = ExplicitTool(parameters=explicit_parameters)

    # Should use explicit parameters, not auto-generated ones
    assert tool.parameters["name"].description == "Explicit name description"
    assert tool.parameters["age"].description == "Explicit age description"
    assert tool.parameters["active"].description == "Explicit active description"


def test_tool_parameter_json_type():
    """Test that ToolParameter correctly converts Python types to JSON types."""

    # Test different types
    str_param = ToolParameter(name="test", type_annotation=str, required=True)
    assert str_param.json_type == "string"

    int_param = ToolParameter(name="test", type_annotation=int, required=True)
    assert int_param.json_type == "integer"

    float_param = ToolParameter(name="test", type_annotation=float, required=True)
    assert float_param.json_type == "number"

    bool_param = ToolParameter(name="test", type_annotation=bool, required=True)
    assert bool_param.json_type == "boolean"

    list_param = ToolParameter(name="test", type_annotation=list, required=True)
    assert list_param.json_type == "array"

    dict_param = ToolParameter(name="test", type_annotation=dict, required=True)
    assert dict_param.json_type == "object"


def test_tool_parameter_union_types():
    """Test that ToolParameter correctly handles union types (e.g., int | None)."""

    # Test int | None (should return "integer")
    int_or_none_param = ToolParameter(
        name="test", type_annotation=int | None, required=True
    )
    assert int_or_none_param.json_type == "integer"

    # Test str | None (should return "string")
    str_or_none_param = ToolParameter(
        name="test", type_annotation=str | None, required=True
    )
    assert str_or_none_param.json_type == "string"

    # Test float | None (should return "number")
    float_or_none_param = ToolParameter(
        name="test", type_annotation=float | None, required=True
    )
    assert float_or_none_param.json_type == "number"

    # Test bool | None (should return "boolean")
    bool_or_none_param = ToolParameter(
        name="test", type_annotation=bool | None, required=True
    )
    assert bool_or_none_param.json_type == "boolean"

    # Test list | None (should return "array")
    list_or_none_param = ToolParameter(
        name="test", type_annotation=list | None, required=True
    )
    assert list_or_none_param.json_type == "array"

    # Test dict | None (should return "object")
    dict_or_none_param = ToolParameter(
        name="test", type_annotation=dict | None, required=True
    )
    assert dict_or_none_param.json_type == "object"


def test_tool_parameter_complex_union_types():
    """Test that ToolParameter correctly handles more complex union types."""

    # Test int | str (should return "integer" - first non-None type)
    int_or_str_param = ToolParameter(
        name="test", type_annotation=int | str, required=True
    )
    assert int_or_str_param.json_type == "integer"

    # Test str | int (should return "string" - first non-None type)
    str_or_int_param = ToolParameter(
        name="test", type_annotation=str | int, required=True
    )
    assert str_or_int_param.json_type == "string"

    # Test float | int | str (should return "number" - first non-None type)
    float_or_int_or_str_param = ToolParameter(
        name="test", type_annotation=float | int | str, required=True
    )
    assert float_or_int_or_str_param.json_type == "number"

    # Test None | int (should return "integer" - first non-None type)
    none_or_int_param = ToolParameter(
        name="test", type_annotation=None | int, required=True
    )
    assert none_or_int_param.json_type == "integer"

    # Test None | int (should return "integer" - first non-None type)
    none_or_int_param = ToolParameter(
        name="test", type_annotation=None | int, required=True
    )
    assert none_or_int_param.json_type == "integer"

    # Test edge case with just None type (should return "string" as fallback)
    none_param = ToolParameter(name="test", type_annotation=type(None), required=True)
    assert none_param.json_type == "string"


def test_tool_parameter_union_types_with_tool_decorator():
    """Test that the @tool decorator correctly handles union types."""

    @tool("A tool with union type parameters")
    def union_type_tool(
        required_int: int | None,
        optional_str: str | None = None,
        mixed_types: int | str = 42,
    ):
        """A tool that uses union types in its parameters."""
        return f"int: {required_int}, str: {optional_str}, mixed: {mixed_types}"

    # Test that the tool can be created without errors
    assert union_type_tool.get_tool_name() == "UnionTypeTool"
    assert union_type_tool.get_tool_description() == "A tool with union type parameters"

    # Test execution with union type parameters
    result = union_type_tool.run(
        required_int=10, optional_str="test", mixed_types="hello"
    )
    assert result == "int: 10, str: test, mixed: hello"

    # Test with None values
    result = union_type_tool.run(required_int=None, optional_str=None, mixed_types=123)
    assert result == "int: None, str: None, mixed: 123"


def test_tool_parameter_union_types_in_base_tool():
    """Test that BaseTool correctly handles union types in parameter extraction."""

    class UnionTypeBaseTool(BaseTool):
        """A base tool that uses union types in its parameters."""

        def execute_tool(
            self,
            user_id: int | None,
            username: str | None = None,
            is_active: bool | None = True,
            environment: dict = None,
            **kwargs,
        ):
            """Execute the tool with union type parameters.

            :param user_id: The user ID (can be None)
            :param username: The username (can be None)
            :param is_active: Whether the user is active (can be None)
            """
            return f"User {user_id} ({username}) - Active: {is_active}"

    tool = UnionTypeBaseTool()

    # Test that parameters are correctly extracted
    assert "user_id" in tool.parameters
    assert "username" in tool.parameters
    assert "is_active" in tool.parameters

    # Test that union types are correctly handled
    assert tool.parameters["user_id"].json_type == "integer"
    assert tool.parameters["username"].json_type == "string"
    assert tool.parameters["is_active"].json_type == "boolean"

    # Test that required/optional status is correct
    assert tool.parameters["user_id"].required
    assert not tool.parameters["username"].required
    assert not tool.parameters["is_active"].required

    # Test execution
    result = tool.run(user_id=123, username="john_doe", is_active=True)
    assert result == "User 123 (john_doe) - Active: True"

    # Test with None values
    result = tool.run(user_id=None, username=None, is_active=None)
    assert result == "User None (None) - Active: None"


def test_tool_parameter_union_types_bedrock_converse():
    """Test that union types are correctly handled in Bedrock Converse format."""

    class UnionTypeBedrockTool(BaseTool):
        """A tool for testing Bedrock format with union types."""

        def execute_tool(
            self,
            query: str | None,
            max_results: int | None = 10,
            environment: dict = None,
            **kwargs,
        ):
            """Execute the tool.

            :param query: The search query (can be None)
            :param max_results: Maximum number of results (can be None)
            """
            return f"Search: {query}, Max: {max_results}"

    tool = UnionTypeBedrockTool()
    bedrock_format = tool.model_dump_bedrock_converse()

    properties = bedrock_format["toolSpec"]["inputSchema"]["json"]["properties"]

    # Check that union types are correctly mapped to JSON schema types
    assert properties["query"]["type"] == "string"
    assert properties["max_results"]["type"] == "integer"

    # Check descriptions
    assert properties["query"]["description"] == "The search query (can be None)"
    assert (
        properties["max_results"]["description"]
        == "Maximum number of results (can be None)"
    )

    # Check required fields
    required = bedrock_format["toolSpec"]["inputSchema"]["json"]["required"]
    assert "query" in required
    assert "max_results" not in required  # Has default value


def test_get_representation_with_unified_parameters():
    """Test that get_representation works with unified parameters."""

    class DocumentedTool(BaseTool):
        """A tool with documented parameters."""

        def execute_tool(
            self,
            name: str,
            age: int,
            active: bool = True,
            environment: dict = None,
            **kwargs,
        ):
            """Execute the tool.

            :param name: The person's name
            :param age: The person's age in years
            :param active: Whether the person is active
            """
            return f"Hello {name}"

    tool = DocumentedTool()
    representation = tool.get_representation()

    # Check that parameter descriptions are included
    assert "name: str (required) - The person's name" in representation
    assert "age: int (required) - The person's age in years" in representation
    assert "active: bool (optional) - Whether the person is active" in representation


def test_bedrock_converse_with_unified_parameters():
    """Test that model_dump_bedrock_converse works with unified parameters."""

    class DocumentedTool(BaseTool):
        """A tool with documented parameters."""

        def execute_tool(
            self, message: str, count: int = 1, environment: dict = None, **kwargs
        ):
            """Execute the tool.

            :param message: The message to process
            :param count: How many times to repeat
            """
            return message * count

    tool = DocumentedTool()
    bedrock_format = tool.model_dump_bedrock_converse()

    properties = bedrock_format["toolSpec"]["inputSchema"]["json"]["properties"]

    assert properties["message"]["description"] == "The message to process"
    assert properties["message"]["type"] == "string"
    assert properties["count"]["description"] == "How many times to repeat"
    assert properties["count"]["type"] == "integer"

    # Check required fields
    required = bedrock_format["toolSpec"]["inputSchema"]["json"]["required"]
    assert "message" in required
    assert "count" not in required  # Has default value


def test_bedrock_converse_fallback_descriptions():
    """Test that model_dump_bedrock_converse falls back to generic descriptions."""

    class UndocumentedTool(BaseTool):
        """A tool without parameter documentation."""

        def execute_tool(self, message: str, environment: dict = None, **kwargs):
            return message

    tool = UndocumentedTool()
    bedrock_format = tool.model_dump_bedrock_converse()

    properties = bedrock_format["toolSpec"]["inputSchema"]["json"]["properties"]

    # Should fall back to generic description
    assert (
        properties["message"]["description"]
        == "Parameter message for the UndocumentedTool tool"
    )


def test_explicit_parameters_in_representation():
    """Test that explicit parameters appear correctly in tool representation."""

    class RepresentationTool(BaseTool):
        """A tool for testing representation with explicit parameters."""

        def execute_tool(
            self, city: str, temperature: float, environment: dict = None, **kwargs
        ):
            return f"Weather in {city}: {temperature}°C"

    explicit_parameters = {
        "city": ToolParameter(
            name="city",
            type_annotation=str,
            description="The name of the city for weather lookup",
            required=True,
        ),
        "temperature": ToolParameter(
            name="temperature",
            type_annotation=float,
            description="Current temperature in Celsius",
            required=True,
        ),
    }

    tool = RepresentationTool(parameters=explicit_parameters)
    representation = tool.get_representation()

    # Check that explicit descriptions are included
    assert (
        "city: str (required) - The name of the city for weather lookup"
        in representation
    )
    assert (
        "temperature: float (required) - Current temperature in Celsius"
        in representation
    )


def test_explicit_parameters_bedrock_converse():
    """Test that explicit parameters are used in Bedrock Converse format."""

    class BedrockTool(BaseTool):
        """A tool for testing Bedrock format with explicit parameters."""

        def execute_tool(
            self, query: str, max_results: int = 10, environment: dict = None, **kwargs
        ):
            return f"Search results for {query}"

    explicit_parameters = {
        "query": ToolParameter(
            name="query",
            type_annotation=str,
            description="The search query string",
            required=True,
        ),
        "max_results": ToolParameter(
            name="max_results",
            type_annotation=int,
            description="Maximum number of results to return",
            required=False,
            default_value=10,
        ),
    }

    tool = BedrockTool(parameters=explicit_parameters)
    bedrock_format = tool.model_dump_bedrock_converse()

    properties = bedrock_format["toolSpec"]["inputSchema"]["json"]["properties"]

    assert properties["query"]["description"] == "The search query string"
    assert properties["query"]["type"] == "string"
    assert (
        properties["max_results"]["description"]
        == "Maximum number of results to return"
    )
    assert properties["max_results"]["type"] == "integer"

    # Check required fields
    required = bedrock_format["toolSpec"]["inputSchema"]["json"]["required"]
    assert "query" in required
    assert "max_results" not in required  # Has default value


def test_parameters_docstring_fallback():
    """Test that tools fall back to docstring parsing when no explicit parameters provided."""

    class FallbackTool(BaseTool):
        """A tool that should fall back to docstring parsing."""

        def execute_tool(self, name: str, age: int, environment: dict = None, **kwargs):
            """Execute the tool.

            :param name: Name from docstring
            :param age: Age from docstring
            """
            return f"Hello {name}"

    # Default behavior should parse docstring
    tool = FallbackTool()

    # Should parse docstring automatically
    assert tool.parameters["name"].description == "Name from docstring"
    assert tool.parameters["age"].description == "Age from docstring"
    assert tool.parameters["name"].required
    assert tool.parameters["age"].required
