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


def test_init(mock_mcp_tool, mock_mcp_client):
    """Test initialization of MCPAgentTool."""
    tool = MCPAgentTool(mock_mcp_tool, mock_mcp_client)
    
    assert tool.mcp_tool == mock_mcp_tool
    assert tool.mcp_client == mock_mcp_client
    assert tool.description == mock_mcp_tool.description


def test_init_without_description(mock_mcp_client):
    """Test initialization when MCPTool has no description."""
    mock_tool = MagicMock(spec=MCPTool)
    mock_tool.name = "test_tool"
    mock_tool.description = None
    
    tool = MCPAgentTool(mock_tool, mock_mcp_client)
    assert tool.description == f"Tool which performs {mock_tool.name}"


def test_get_tool_name(mcp_agent_tool, mock_mcp_tool):
    """Test get_tool_name method."""
    assert mcp_agent_tool.get_tool_name() == mock_mcp_tool.name


def test_get_tool_signature(mcp_agent_tool):
    """Test get_tool_signature method."""
    signature = mcp_agent_tool.get_tool_signature()
    
    # Check that we have the correct number of parameters
    assert len(signature) == 7  # param1, param2, days, bool_param, array_param, object_param, and kwargs
    
    # Check required parameter
    param1 = signature["param1"]
    assert param1.annotation == str
    assert param1.default == inspect.Parameter.empty
    
    # Check optional parameter
    param2 = signature["param2"]
    assert param2.annotation == int
    assert param2.default is None
    
    # Check boolean parameter
    bool_param = signature["bool_param"]
    assert bool_param.annotation == bool
    assert bool_param.default is None
    
    # Check array parameter
    array_param = signature["array_param"]
    assert array_param.annotation == str  # Default to str for unknown types
    assert array_param.default is None
    
    # Check object parameter
    object_param = signature["object_param"]
    assert object_param.annotation == str  # Default to str for unknown types
    assert object_param.default is None
    
    # Check kwargs parameter
    kwargs = signature["kwargs"]
    assert kwargs.kind == inspect.Parameter.VAR_KEYWORD


def test_get_tool_signature_empty_schema(mock_mcp_client):
    """Test get_tool_signature with empty schema."""
    mock_tool = MagicMock(spec=MCPTool)
    mock_tool.name = "test_tool"
    mock_tool.description = "Test tool description"
    mock_tool.inputSchema = {
        "type": "object",
        "properties": {},
        "required": []
    }
    
    tool = MCPAgentTool(mock_tool, mock_mcp_client)
    signature = tool.get_tool_signature()
    
    assert len(signature) == 1  # Only kwargs
    assert "kwargs" in signature
    assert signature["kwargs"].kind == inspect.Parameter.VAR_KEYWORD


def test_get_tool_signature_missing_properties(mock_mcp_client):
    """Test get_tool_signature with missing properties in schema."""
    mock_tool = MagicMock(spec=MCPTool)
    mock_tool.name = "test_tool"
    mock_tool.description = "Test tool description"
    mock_tool.inputSchema = {
        "type": "object",
        "required": []
    }
    
    tool = MCPAgentTool(mock_tool, mock_mcp_client)
    signature = tool.get_tool_signature()
    
    assert len(signature) == 1  # Only kwargs
    assert "kwargs" in signature


def test_get_tool_params(mcp_agent_tool):
    """Test get_tool_params method."""
    params = mcp_agent_tool.get_tool_params()
    assert params == mcp_agent_tool.get_tool_signature()


def test_model_dump_bedrock_converse(mcp_agent_tool, mock_mcp_tool):
    """Test model_dump_bedrock_converse method."""
    result = mcp_agent_tool.model_dump_bedrock_converse()
    
    expected = {
        "toolSpec": {
            "name": mock_mcp_tool.name,
            "description": mock_mcp_tool.description,
            "inputSchema": {
                "json": mock_mcp_tool.inputSchema
            }
        }
    }
    
    assert result == expected


@patch('os.urandom')
def test_execute_tool(mock_urandom, mcp_agent_tool, mock_mcp_client):
    """Test execute_tool method."""
    # Setup
    mock_urandom.return_value = b'\x01\x02\x03\x04'
    test_args = {"param1": "value1", "param2": 42}
    expected_tool_use_id = "mcp_test_tool_01020304"
    mock_mcp_client.call_tool_sync.return_value = "test_response"
    
    # Execute
    result = mcp_agent_tool.execute_tool(**test_args)
    
    # Verify
    mock_mcp_client.call_tool_sync.assert_called_once_with(
        tool_use_id=expected_tool_use_id,
        name="test_tool",
        arguments=test_args
    )
    assert result == "test_response"


def test_execute_tool_with_custom_tool_use_id(mcp_agent_tool, mock_mcp_client):
    """Test execute_tool method with custom tool_use_id."""
    test_args = {
        "param1": "value1",
        "tool_use_id": "custom_id"
    }
    mock_mcp_client.call_tool_sync.return_value = "test_response"
    
    result = mcp_agent_tool.execute_tool(**test_args)
    
    # Verify tool_use_id was removed from arguments
    expected_args = {"param1": "value1"}
    mock_mcp_client.call_tool_sync.assert_called_once_with(
        tool_use_id="custom_id",
        name="test_tool",
        arguments=expected_args
    )
    assert result == "test_response"


def test_execute_tool_with_environment(mcp_agent_tool, mock_mcp_client):
    """Test execute_tool method with environment parameter."""
    test_args = {"param1": "value1"}
    environment = {"env_var": "value"}
    mock_mcp_client.call_tool_sync.return_value = "test_response"
    
    result = mcp_agent_tool.execute_tool(environment=environment, **test_args)
    
    mock_mcp_client.call_tool_sync.assert_called_once()
    assert result == "test_response"


def test_execute_tool_with_empty_environment(mcp_agent_tool, mock_mcp_client):
    """Test execute_tool method with empty environment."""
    test_args = {"param1": "value1"}
    environment = {}
    mock_mcp_client.call_tool_sync.return_value = "test_response"
    
    result = mcp_agent_tool.execute_tool(environment=environment, **test_args)
    
    mock_mcp_client.call_tool_sync.assert_called_once()
    assert result == "test_response"


def test_execute_tool_with_none_environment(mcp_agent_tool, mock_mcp_client):
    """Test execute_tool method with None environment."""
    test_args = {"param1": "value1"}
    mock_mcp_client.call_tool_sync.return_value = "test_response"
    
    result = mcp_agent_tool.execute_tool(environment=None, **test_args)
    
    mock_mcp_client.call_tool_sync.assert_called_once()
    assert result == "test_response"


def test_execute_tool_with_error(mcp_agent_tool, mock_mcp_client):
    """Test execute_tool method when client raises an error."""
    test_args = {"param1": "value1"}
    mock_mcp_client.call_tool_sync.side_effect = Exception("Test error")
    
    with pytest.raises(Exception) as exc_info:
        mcp_agent_tool.execute_tool(**test_args)
    
    assert str(exc_info.value) == "Test error"
    mock_mcp_client.call_tool_sync.assert_called_once() 
