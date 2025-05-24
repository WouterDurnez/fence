import inspect
import os
from unittest.mock import MagicMock, patch

import pytest
from mcp.types import Tool as MCPTool

from fence.tools.mcp.mcp_tool import MCPAgentTool


@pytest.fixture
def mock_mcp_tool():
    """Fixture to create a mock MCPTool instance."""
    mock_tool = MagicMock(spec=MCPTool)
    mock_tool.name = "test_tool"
    mock_tool.description = "Test tool description"
    mock_tool.inputSchema = {
        "type": "object",
        "properties": {
            "param1": {"type": "string"},
            "param2": {"type": "integer"},
            "days": {"type": "number"},
            "bool_param": {"type": "boolean"},
            "array_param": {"type": "array"},
            "object_param": {"type": "object"}
        },
        "required": ["param1"]
    }
    return mock_tool


@pytest.fixture
def mock_mcp_client():
    """Fixture to create a mock MCPClient instance."""
    return MagicMock()


@pytest.fixture
def mcp_agent_tool(mock_mcp_tool, mock_mcp_client):
    """Fixture to create an MCPAgentTool instance with mocked dependencies."""
    return MCPAgentTool(mock_mcp_tool, mock_mcp_client)


# ... [previous test cases remain unchanged] ...


@pytest.mark.parametrize("schema,expected_params", [
    # Test case 1: Empty schema
    (
        {
            "type": "object",
            "properties": {},
            "required": []
        },
        {
            "kwargs": inspect.Parameter("kwargs", inspect.Parameter.VAR_KEYWORD)
        }
    ),
    
    # Test case 2: Schema with only required parameters
    (
        {
            "type": "object",
            "properties": {
                "required_str": {"type": "string"},
                "required_int": {"type": "integer"}
            },
            "required": ["required_str", "required_int"]
        },
        {
            "required_str": inspect.Parameter("required_str", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=str),
            "required_int": inspect.Parameter("required_int", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=int),
            "kwargs": inspect.Parameter("kwargs", inspect.Parameter.VAR_KEYWORD)
        }
    ),
    
    # Test case 3: Schema with only optional parameters
    (
        {
            "type": "object",
            "properties": {
                "optional_str": {"type": "string"},
                "optional_int": {"type": "integer"}
            },
            "required": []
        },
        {
            "optional_str": inspect.Parameter("optional_str", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=str, default=None),
            "optional_int": inspect.Parameter("optional_int", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=int, default=None),
            "kwargs": inspect.Parameter("kwargs", inspect.Parameter.VAR_KEYWORD)
        }
    ),
    
    # Test case 4: Schema with nested objects
    (
        {
            "type": "object",
            "properties": {
                "nested": {
                    "type": "object",
                    "properties": {
                        "field": {"type": "string"}
                    }
                }
            },
            "required": []
        },
        {
            "nested": inspect.Parameter("nested", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=str, default=None),
            "kwargs": inspect.Parameter("kwargs", inspect.Parameter.VAR_KEYWORD)
        }
    ),
    
    # Test case 5: Schema with array types
    (
        {
            "type": "object",
            "properties": {
                "string_array": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "number_array": {
                    "type": "array",
                    "items": {"type": "number"}
                }
            },
            "required": []
        },
        {
            "string_array": inspect.Parameter("string_array", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=str, default=None),
            "number_array": inspect.Parameter("number_array", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=str, default=None),
            "kwargs": inspect.Parameter("kwargs", inspect.Parameter.VAR_KEYWORD)
        }
    ),
    
    # Test case 6: Schema with unknown types
    (
        {
            "type": "object",
            "properties": {
                "unknown_type": {"type": "unknown"},
                "missing_type": {}
            },
            "required": []
        },
        {
            "unknown_type": inspect.Parameter("unknown_type", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=str, default=None),
            "missing_type": inspect.Parameter("missing_type", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=str, default=None),
            "kwargs": inspect.Parameter("kwargs", inspect.Parameter.VAR_KEYWORD)
        }
    ),
    
    # Test case 7: Schema with special case for 'days' parameter
    (
        {
            "type": "object",
            "properties": {
                "days": {"type": "number"},
                "other_days": {"type": "number"}
            },
            "required": []
        },
        {
            "days": inspect.Parameter("days", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=float, default=7),
            "other_days": inspect.Parameter("other_days", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=float, default=None),
            "kwargs": inspect.Parameter("kwargs", inspect.Parameter.VAR_KEYWORD)
        }
    ),
    
    # Test case 8: Schema with mixed required and optional parameters
    (
        {
            "type": "object",
            "properties": {
                "required_str": {"type": "string"},
                "optional_str": {"type": "string"},
                "required_int": {"type": "integer"},
                "optional_int": {"type": "integer"}
            },
            "required": ["required_str", "required_int"]
        },
        {
            "required_str": inspect.Parameter("required_str", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=str),
            "optional_str": inspect.Parameter("optional_str", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=str, default=None),
            "required_int": inspect.Parameter("required_int", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=int),
            "optional_int": inspect.Parameter("optional_int", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=int, default=None),
            "kwargs": inspect.Parameter("kwargs", inspect.Parameter.VAR_KEYWORD)
        }
    ),
    
    # Test case 9: Schema with missing properties
    (
        {
            "type": "object",
            "required": []
        },
        {
            "kwargs": inspect.Parameter("kwargs", inspect.Parameter.VAR_KEYWORD)
        }
    ),
    
    # Test case 10: Schema with missing required
    (
        {
            "type": "object",
            "properties": {
                "param": {"type": "string"}
            }
        },
        {
            "param": inspect.Parameter("param", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=str, default=None),
            "kwargs": inspect.Parameter("kwargs", inspect.Parameter.VAR_KEYWORD)
        }
    )
])
def test_get_tool_signature_parameterized(mock_mcp_client, schema, expected_params):
    """Test get_tool_signature with various schema configurations."""
    mock_tool = MagicMock(spec=MCPTool)
    mock_tool.name = "test_tool"
    mock_tool.inputSchema = schema
    
    tool = MCPAgentTool(mock_tool, mock_mcp_client)
    signature = tool.get_tool_signature()
    
    # Check number of parameters
    assert len(signature) == len(expected_params)
    
    # Check each parameter
    for name, expected_param in expected_params.items():
        assert name in signature
        actual_param = signature[name]
        
        # Check parameter kind
        assert actual_param.kind == expected_param.kind
        
        # Check parameter annotation
        if expected_param.annotation != inspect.Parameter.empty:
            assert actual_param.annotation == expected_param.annotation
        
        # Check parameter default
        if expected_param.default != inspect.Parameter.empty:
            assert actual_param.default == expected_param.default


def test_get_tool_signature_with_complex_nested_schema(mock_mcp_client):
    """Test get_tool_signature with a complex nested schema."""
    mock_tool = MagicMock(spec=MCPTool)
    mock_tool.name = "test_tool"
    mock_tool.inputSchema = {
        "type": "object",
        "properties": {
            "nested_object": {
                "type": "object",
                "properties": {
                    "nested_string": {"type": "string"},
                    "nested_number": {"type": "number"}
                }
            },
            "array_of_objects": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "value": {"type": "number"}
                    }
                }
            },
            "mixed_array": {
                "type": "array",
                "items": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "number"}
                    ]
                }
            }
        },
        "required": ["nested_object"]
    }
    
    tool = MCPAgentTool(mock_tool, mock_mcp_client)
    signature = tool.get_tool_signature()
    
    # Should only include top-level properties
    assert "nested_object" in signature
    assert "array_of_objects" in signature
    assert "mixed_array" in signature
    assert "kwargs" in signature
    
    # Check that nested properties are not included
    assert "nested_string" not in signature
    assert "nested_number" not in signature
    assert "name" not in signature
    assert "value" not in signature
    
    # Check parameter types (should default to str for complex types)
    assert signature["nested_object"].annotation == str
    assert signature["array_of_objects"].annotation == str
    assert signature["mixed_array"].annotation == str
    
    # Check required parameter has no default
    assert signature["nested_object"].default == inspect.Parameter.empty
    
    # Check optional parameters have None default
    assert signature["array_of_objects"].default is None
    assert signature["mixed_array"].default is None


def test_get_tool_signature_with_invalid_schema(mock_mcp_client):
    """Test get_tool_signature with invalid schema structures."""
    mock_tool = MagicMock(spec=MCPTool)
    mock_tool.name = "test_tool"
    
    # Test with invalid schema type
    mock_tool.inputSchema = {
        "type": "invalid_type",
        "properties": {
            "param": {"type": "string"}
        }
    }
    
    tool = MCPAgentTool(mock_tool, mock_mcp_client)
    signature = tool.get_tool_signature()
    
    # Should still return kwargs parameter
    assert len(signature) == 1
    assert "kwargs" in signature
    assert signature["kwargs"].kind == inspect.Parameter.VAR_KEYWORD
    
    # Test with missing type
    mock_tool.inputSchema = {
        "properties": {
            "param": {"type": "string"}
        }
    }
    
    tool = MCPAgentTool(mock_tool, mock_mcp_client)
    signature = tool.get_tool_signature()
    
    # Should still return kwargs parameter
    assert len(signature) == 1
    assert "kwargs" in signature
    assert signature["kwargs"].kind == inspect.Parameter.VAR_KEYWORD 