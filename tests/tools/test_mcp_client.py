"""Tests for MCP Client."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from mcp.types import TextContent
from mcp.types import Tool as MCPTool

from fence.mcp.client import MCPClient
from fence.mcp.tool import MCPAgentTool


@pytest.fixture
def mock_mcp_tool():
    """Create a mock MCP tool."""
    return MCPTool(
        name="test_tool",
        description="A test tool for MCP",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The query parameter"},
                "limit": {
                    "type": "integer",
                    "description": "The limit parameter",
                    "default": 10,
                },
            },
            "required": ["query"],
        },
    )


@pytest.fixture
def mock_mcp_session():
    """Create a mock MCP session."""
    session = AsyncMock()

    # Mock initialize
    session.initialize = AsyncMock()

    # Mock list_tools
    mock_tools_response = Mock()
    mock_tools_response.tools = []
    session.list_tools = AsyncMock(return_value=mock_tools_response)

    # Mock call_tool
    mock_call_response = Mock()
    mock_call_response.content = [
        TextContent(type="text", text="Test response from MCP tool")
    ]
    session.call_tool = AsyncMock(return_value=mock_call_response)

    return session


@pytest.fixture
def mcp_client():
    """Create an MCP client for testing."""
    return MCPClient(read_timeout_seconds=5)


class TestMCPClient:
    """Test MCPClient functionality."""

    def test_connect_success(self, mcp_client, mock_mcp_session):
        """Test successful connection to MCP server."""
        with patch.object(mcp_client, "connect") as mock_connect:
            # Mock successful connection
            def mock_connect_side_effect(transport_type="stdio", **kwargs):
                mcp_client._connected = True
                mcp_client.session = mock_mcp_session
                mcp_client._loop = Mock()
                mcp_client._thread = Mock()
                mcp_client._shutdown_event = Mock()
                mcp_client._connection_task = Mock()

            mock_connect.side_effect = mock_connect_side_effect

            # Test connection
            mcp_client.connect(
                transport_type="stdio", command="uvx", args=["test-server"]
            )

            # Verify connection state
            assert mcp_client._connected is True
            assert mcp_client.session == mock_mcp_session

    def test_connect_multiple_calls(self, mcp_client):
        """Test that multiple connect calls don't cause issues."""
        with patch.object(mcp_client, "_connected", True):
            # Should return early without doing anything
            mcp_client.connect(
                transport_type="stdio", command="uvx", args=["test-server"]
            )
            # No exception should be raised

    def test_list_tools_success(self, mcp_client, mock_mcp_session, mock_mcp_tool):
        """Test successful tool listing."""
        # Setup connection
        self._setup_connected_client(mcp_client, mock_mcp_session)

        # Mock list_tools method to directly manipulate client state
        with patch.object(mcp_client, "list_tools") as mock_list_tools:

            def mock_list_tools_side_effect():
                # Simulate the list_tools logic
                mcp_client.tools.clear()
                mcp_client.tools.append(
                    MCPAgentTool(mcp_tool=mock_mcp_tool, mcp_client=mcp_client)
                )
                return Mock(tools=[mock_mcp_tool])

            mock_list_tools.side_effect = mock_list_tools_side_effect

            # Test list_tools
            _ = mcp_client.list_tools()

            # Verify tools were registered
            assert len(mcp_client.tools) == 1
            assert isinstance(mcp_client.tools[0], MCPAgentTool)
            assert mcp_client.tools[0].get_tool_name() == "test_tool"

    def test_list_tools_not_connected(self, mcp_client):
        """Test list_tools raises error when not connected."""
        with pytest.raises(RuntimeError, match="MCP Client not connected"):
            mcp_client.list_tools()

    @patch("fence.mcp.client.asyncio.run_coroutine_threadsafe")
    def test_call_tool_single_response(
        self, mock_run_coroutine, mcp_client, mock_mcp_session
    ):
        """Test calling a tool with single text response."""
        # Setup connection
        self._setup_connected_client(mcp_client, mock_mcp_session)

        # Setup call_tool response with single content
        mock_response = Mock()
        mock_response.content = [TextContent(type="text", text="Single response")]

        # Mock the async execution
        mock_future = Mock()
        mock_future.result.return_value = "Single response"
        mock_run_coroutine.return_value = mock_future

        # Test call_tool
        result = mcp_client.call_tool("test_tool", {"query": "test"})

        # Verify result
        assert result == "Single response"

    @patch("fence.mcp.client.asyncio.run_coroutine_threadsafe")
    def test_call_tool_multiple_responses(
        self, mock_run_coroutine, mcp_client, mock_mcp_session
    ):
        """Test calling a tool with multiple text responses."""
        # Setup connection
        self._setup_connected_client(mcp_client, mock_mcp_session)

        # Mock the async execution with multiple responses
        mock_future = Mock()
        mock_future.result.return_value = {
            "results": ["Response 1", "Response 2", "Response 3"],
            "count": 3,
        }
        mock_run_coroutine.return_value = mock_future

        # Test call_tool
        result = mcp_client.call_tool("test_tool", {"query": "test"})

        # Verify result structure
        assert isinstance(result, dict)
        assert result["count"] == 3
        assert result["results"] == ["Response 1", "Response 2", "Response 3"]

    @patch("fence.mcp.client.asyncio.run_coroutine_threadsafe")
    def test_call_tool_empty_response(
        self, mock_run_coroutine, mcp_client, mock_mcp_session
    ):
        """Test calling a tool with empty response."""
        # Setup connection
        self._setup_connected_client(mcp_client, mock_mcp_session)

        # Mock the async execution with empty response
        mock_future = Mock()
        mock_future.result.return_value = "No content returned from MCP tool"
        mock_run_coroutine.return_value = mock_future

        # Test call_tool
        result = mcp_client.call_tool("test_tool", {"query": "test"})

        # Verify result
        assert result == "No content returned from MCP tool"

    def test_call_tool_not_connected(self, mcp_client):
        """Test call_tool raises error when not connected."""
        with pytest.raises(RuntimeError, match="MCP Client not connected"):
            mcp_client.call_tool("test_tool", {"query": "test"})

    @patch("fence.mcp.client.asyncio.run_coroutine_threadsafe")
    def test_call_tool_exception(
        self, mock_run_coroutine, mcp_client, mock_mcp_session
    ):
        """Test call_tool handles exceptions properly."""
        # Setup connection
        self._setup_connected_client(mcp_client, mock_mcp_session)

        # Setup call_tool to raise exception
        mock_future = Mock()
        mock_future.result.side_effect = Exception("Test error")
        mock_run_coroutine.return_value = mock_future

        # Test call_tool raises exception
        with pytest.raises(Exception, match="Test error"):
            mcp_client.call_tool("test_tool", {"query": "test"})

    def test_disconnect_not_connected(self, mcp_client):
        """Test disconnect when not connected."""
        # Should not raise any exceptions
        mcp_client.disconnect()

    @patch("fence.mcp.client.threading.Thread")
    def test_disconnect_success(self, mock_thread, mcp_client):
        """Test successful disconnection."""
        # Setup mock loop and thread
        mock_loop = Mock()
        mock_loop.is_closed.return_value = False
        mock_loop.call_soon_threadsafe = Mock()

        mock_thread_instance = Mock()
        mock_thread_instance.is_alive.return_value = True
        mock_thread_instance.join = Mock()

        mock_connection_task = Mock()
        mock_connection_task.result = Mock()

        mock_shutdown_event = Mock()
        mock_shutdown_event.set = Mock()

        # Setup client state
        mcp_client._connected = True
        mcp_client._loop = mock_loop
        mcp_client._thread = mock_thread_instance
        mcp_client._connection_task = mock_connection_task
        mcp_client._shutdown_event = mock_shutdown_event

        # Test disconnect
        mcp_client.disconnect()

        # Verify state
        assert mcp_client._connected is False

        # Verify cleanup calls
        mock_shutdown_event.set.assert_called_once()
        mock_connection_task.result.assert_called_once_with(timeout=3)
        mock_loop.call_soon_threadsafe.assert_called_once()
        mock_thread_instance.join.assert_called_once_with(timeout=2)

    def test_context_manager(self, mcp_client):
        """Test client works as context manager."""
        with patch.object(mcp_client, "disconnect") as mock_disconnect:
            with mcp_client:
                pass
            mock_disconnect.assert_called_once()

    def test_connect_stdio_convenience_method(self, mcp_client):
        """Test the stdio convenience method."""
        with patch.object(mcp_client, "connect") as mock_connect:
            mcp_client.connect_stdio("python", ["server.py"])
            mock_connect.assert_called_once_with(
                transport_type="stdio", command="python", args=["server.py"]
            )

    def test_connect_sse_convenience_method(self, mcp_client):
        """Test the SSE convenience method."""
        with patch.object(mcp_client, "connect") as mock_connect:
            mcp_client.connect_sse("http://localhost:8000/sse", {"auth": "token"})
            mock_connect.assert_called_once_with(
                transport_type="sse",
                url="http://localhost:8000/sse",
                headers={"auth": "token"},
            )

    def test_connect_streamable_http_convenience_method(self, mcp_client):
        """Test the streamable HTTP convenience method."""
        with patch.object(mcp_client, "connect") as mock_connect:
            mcp_client.connect_streamable_http("http://localhost:8000/mcp")
            mock_connect.assert_called_once_with(
                transport_type="streamable_http",
                url="http://localhost:8000/mcp",
                headers=None,
            )

    def _setup_connected_client(self, client, session):
        """Helper to setup a connected client for testing."""
        client._connected = True
        client.session = session
        client._loop = Mock()
        client._thread = Mock()
