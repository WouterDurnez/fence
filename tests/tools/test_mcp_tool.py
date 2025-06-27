"""Tests for MCP Agent Tool."""

from unittest.mock import Mock

import pytest
from mcp.types import Tool as MCPTool

from fence.mcp.tool import MCPAgentTool
from fence.tools.base import ToolParameter


@pytest.fixture
def mock_mcp_tool():
    """Create a mock MCP tool."""
    return MCPTool(
        name="search_docs",
        description="Search documentation",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "limit": {
                    "type": "integer",
                    "description": "Maximum results",
                    "default": 10,
                },
                "include_archived": {
                    "type": "boolean",
                    "description": "Include archived results",
                },
            },
            "required": ["query"],
        },
    )


@pytest.fixture
def mock_mcp_client():
    """Create a mock MCP client."""
    client = Mock()
    client.call_tool = Mock(return_value="Mock result")
    return client


@pytest.fixture
def mcp_agent_tool(mock_mcp_tool, mock_mcp_client):
    """Create an MCPAgentTool for testing."""
    return MCPAgentTool(mcp_tool=mock_mcp_tool, mcp_client=mock_mcp_client)


class TestMCPAgentTool:
    """Test MCPAgentTool functionality."""

    def test_initialization(self, mcp_agent_tool, mock_mcp_tool, mock_mcp_client):
        """Test MCPAgentTool initialization."""
        assert mcp_agent_tool.mcp_tool == mock_mcp_tool
        assert mcp_agent_tool.mcp_client == mock_mcp_client
        assert mcp_agent_tool.description == "Search documentation"

    def test_initialization_no_description(self, mock_mcp_client):
        """Test MCPAgentTool initialization when MCP tool has no description."""
        mock_tool = MCPTool(
            name="test_tool",
            description=None,
            inputSchema={"type": "object", "properties": {}},
        )

        agent_tool = MCPAgentTool(mcp_tool=mock_tool, mcp_client=mock_mcp_client)
        assert agent_tool.description == "Tool which performs test_tool"

    def test_build_mcp_parameters(self, mcp_agent_tool):
        """Test parameter building from MCP tool schema."""
        params = mcp_agent_tool.parameters

        # Check required parameter
        assert "query" in params
        query_param = params["query"]
        assert isinstance(query_param, ToolParameter)
        assert query_param.name == "query"
        assert query_param.type_annotation == str
        assert query_param.description == "Search query"
        assert query_param.required is True
        assert query_param.default_value is None

        # Check optional parameter with default
        assert "limit" in params
        limit_param = params["limit"]
        assert limit_param.type_annotation == int
        assert limit_param.required is False
        assert limit_param.default_value == 10

        # Check boolean parameter
        assert "include_archived" in params
        archived_param = params["include_archived"]
        assert archived_param.type_annotation == bool
        assert archived_param.required is False

    def test_get_tool_name(self, mcp_agent_tool):
        """Test getting tool name."""
        assert mcp_agent_tool.get_tool_name() == "search_docs"

    def test_get_tool_params(self, mcp_agent_tool):
        """Test getting tool parameters as inspect.Parameter objects."""
        params = mcp_agent_tool.get_tool_params()

        # Should have the defined parameters plus kwargs
        param_names = list(params.keys())
        assert "query" in param_names
        assert "limit" in param_names
        assert "include_archived" in param_names
        assert "kwargs" in param_names

        # Check required parameter
        query_param = params["query"]
        assert query_param.annotation == str
        assert query_param.default == query_param.empty  # No default for required param

        # Check optional parameter
        limit_param = params["limit"]
        assert limit_param.annotation == int | None
        assert limit_param.default == 10

    def test_execute_tool_success(self, mcp_agent_tool, mock_mcp_client):
        """Test successful tool execution."""
        # Setup mock client response
        mock_mcp_client.call_tool.return_value = "Search results"

        # Execute tool
        result = mcp_agent_tool.execute_tool(query="test query", limit=5)

        # Verify result
        assert result == "Search results"

        # Verify client was called correctly
        mock_mcp_client.call_tool.assert_called_once_with(
            name="search_docs", arguments={"query": "test query", "limit": 5}
        )

    def test_execute_tool_filters_parameters(self, mcp_agent_tool, mock_mcp_client):
        """Test that execute_tool filters out undefined parameters."""
        mock_mcp_client.call_tool.return_value = "Results"

        # Execute with extra parameters that aren't in schema
        _ = mcp_agent_tool.execute_tool(
            query="test",
            limit=5,
            undefined_param="should_be_filtered",
            another_undefined="also_filtered",
        )

        # Verify only defined parameters were passed
        mock_mcp_client.call_tool.assert_called_once_with(
            name="search_docs", arguments={"query": "test", "limit": 5}
        )

    def test_execute_tool_with_environment(self, mcp_agent_tool, mock_mcp_client):
        """Test tool execution with environment parameter."""
        mock_mcp_client.call_tool.return_value = "Results"

        # Execute with environment (should be ignored for MCP tools)
        _ = mcp_agent_tool.execute_tool(environment={"API_KEY": "secret"}, query="test")

        # Verify call
        mock_mcp_client.call_tool.assert_called_once_with(
            name="search_docs", arguments={"query": "test"}
        )

    def test_execute_tool_exception(self, mcp_agent_tool, mock_mcp_client):
        """Test tool execution when client raises exception."""
        # Setup mock to raise exception
        mock_mcp_client.call_tool.side_effect = Exception("Connection error")

        # Execute tool
        result = mcp_agent_tool.execute_tool(query="test")

        # Verify error response
        assert isinstance(result, dict)
        assert "error" in result
        assert "Failed to execute MCP tool 'search_docs'" in result["error"]
        assert "Connection error" in result["error"]

    def test_parameter_type_mapping(self, mock_mcp_client):
        """Test mapping of JSON Schema types to Python types."""
        schema = {
            "type": "object",
            "properties": {
                "string_param": {"type": "string"},
                "int_param": {"type": "integer"},
                "bool_param": {"type": "boolean"},
                "float_param": {"type": "number"},
                "list_param": {"type": "array"},
                "dict_param": {"type": "object"},
                "unknown_param": {"type": "unknown_type"},
            },
            "required": [],
        }

        mock_tool = MCPTool(
            name="type_test", description="Test type mapping", inputSchema=schema
        )

        agent_tool = MCPAgentTool(mcp_tool=mock_tool, mcp_client=mock_mcp_client)
        params = agent_tool.parameters

        # Verify type mappings
        assert params["string_param"].type_annotation == str
        assert params["int_param"].type_annotation == int
        assert params["bool_param"].type_annotation == bool
        assert params["float_param"].type_annotation == float
        assert params["list_param"].type_annotation == list
        assert params["dict_param"].type_annotation == dict
        assert params["unknown_param"].type_annotation == str  # Default to str

    def test_empty_schema(self, mock_mcp_client):
        """Test handling of tools with empty or minimal schema."""
        mock_tool = MCPTool(
            name="minimal_tool",
            description="Minimal tool",
            inputSchema={"type": "object"},  # No properties or required
        )

        agent_tool = MCPAgentTool(mcp_tool=mock_tool, mcp_client=mock_mcp_client)

        # Should have no parameters except kwargs
        tool_params = agent_tool.get_tool_params()
        assert len(tool_params) == 1  # Only kwargs
        assert "kwargs" in tool_params
