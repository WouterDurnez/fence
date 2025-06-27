import asyncio
import concurrent.futures
import logging
import threading
from datetime import timedelta
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import TextContent

from fence.mcp.tool import MCPAgentTool
from fence.tools.base import BaseTool

logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class MCPClient:
    """Synchronous MCP Client that supports multiple transport types with a persistent event loop in a background thread."""

    def __init__(
        self,
        read_timeout_seconds: int = 30,
        transport_type: str = "stdio",
        **transport_kwargs,
    ):
        self.session: ClientSession | None = None
        self.tools: list[BaseTool] = []
        self.read_timeout_seconds = read_timeout_seconds
        self._connected = False
        self._loop = None
        self._thread = None
        self._shutdown_event = None
        self._connection_task = None
        self._transport_type = transport_type
        self._transport_kwargs = transport_kwargs

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure proper cleanup."""
        self.disconnect()

    def connect(self, transport_type: str = None, **transport_kwargs):
        """Connect to the MCP server using the specified transport type.

        :param transport_type: Type of transport ('stdio', 'sse', 'streamable_http')
        :param transport_kwargs: Transport-specific connection parameters

        For stdio transport:
            - command: str - The command to execute
            - args: list[str] - Arguments to pass to the command

        For sse/streamable_http transports:
            - url: str - The endpoint URL
            - headers: dict[str, str] | None - Optional HTTP headers
        """
        if self._connected:
            return

        # Use stored parameters if none provided
        if transport_type is None:
            transport_type = self._transport_type
        if not transport_kwargs:
            transport_kwargs = self._transport_kwargs

        self._shutdown_event = threading.Event()

        # Start event loop in background thread
        def run_event_loop():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            try:
                self._loop.run_forever()
            finally:
                self._loop.close()

        self._thread = threading.Thread(target=run_event_loop, daemon=True)
        self._thread.start()

        # Wait for loop to be ready
        while self._loop is None:
            threading.Event().wait(0.01)

        # Create a single long-running task that manages the entire connection lifecycle
        async def _connection_lifecycle():
            transport = None
            session = None
            try:
                # Create transport based on type
                if transport_type == "stdio":
                    server_params = StdioServerParameters(
                        command=transport_kwargs.get("command"),
                        args=transport_kwargs.get("args", []),
                    )
                    transport = stdio_client(server_params)

                elif transport_type == "sse":
                    transport = sse_client(
                        transport_kwargs.get("url"),
                        headers=transport_kwargs.get("headers"),
                    )

                elif transport_type == "streamable_http":
                    transport = streamablehttp_client(
                        transport_kwargs.get("url"),
                        headers=transport_kwargs.get("headers"),
                    )

                else:
                    raise ValueError(f"Unsupported transport type: {transport_type}")

                # Use proper async context managers with explicit error handling
                try:
                    transport_context = transport.__aenter__()
                    transport_result = await transport_context

                    # Handle different transport types that may return different numbers of values
                    if transport_type == "streamable_http":
                        # streamable_http may return more than 2 values, we need the first two
                        if (
                            isinstance(transport_result, (list, tuple))
                            and len(transport_result) >= 2
                        ):
                            read_stream, write_stream = (
                                transport_result[0],
                                transport_result[1],
                            )
                        else:
                            raise ValueError(
                                f"Unexpected transport result format: {transport_result}"
                            )
                    else:
                        # stdio and sse transports return exactly 2 values
                        read_stream, write_stream = transport_result

                    session_context = ClientSession(
                        read_stream, write_stream
                    ).__aenter__()
                    session = await session_context

                    await session.initialize()
                    self.session = session
                    self._connected = True
                    logger.info(
                        f"MCP server connected using {transport_type} transport"
                    )

                    # Wait for shutdown signal
                    while not self._shutdown_event.is_set():
                        await asyncio.sleep(0.1)

                except Exception as e:
                    logger.error(f"Error during session initialization: {e}")
                    raise
                finally:
                    # Clean up session and transport in reverse order
                    if session:
                        try:
                            await session.__aexit__(None, None, None)
                        except Exception as e:
                            logger.warning(f"Error closing session: {e}")

                    if transport:
                        try:
                            await transport.__aexit__(None, None, None)
                        except Exception as e:
                            logger.warning(f"Error closing transport: {e}")

            except Exception as e:
                logger.error(f"Error in MCP client lifecycle: {e}")
                raise
            finally:
                self._connected = False
                self.session = None

        # Start the lifecycle task
        self._connection_task = asyncio.run_coroutine_threadsafe(
            _connection_lifecycle(), self._loop
        )

        # Wait for connection to be established
        timeout = 30
        while not self._connected and timeout > 0:
            threading.Event().wait(0.1)
            timeout -= 0.1

        if not self._connected:
            raise RuntimeError("Failed to connect to MCP server within timeout")

    def connect_stdio(self, command: str, args: list[str]):
        """Convenience method for stdio connections (backward compatibility).

        :param command: The command to execute
        :param args: Arguments to pass to the command
        """
        self.connect(transport_type="stdio", command=command, args=args)

    def connect_sse(self, url: str, headers: dict[str, str] | None = None):
        """Convenience method for SSE connections.

        :param url: The SSE endpoint URL (typically ending in /sse)
        :param headers: Optional HTTP headers to include
        """
        self.connect(transport_type="sse", url=url, headers=headers)

    def connect_streamable_http(self, url: str, headers: dict[str, str] | None = None):
        """Convenience method for streamable HTTP connections.

        :param url: The HTTP endpoint URL (typically ending in /mcp)
        :param headers: Optional HTTP headers to include
        """
        self.connect(transport_type="streamable_http", url=url, headers=headers)

    def call_tool(self, name: str, arguments: dict[str, Any]) -> str | dict:
        """Call an MCP tool synchronously."""
        if not self._connected:
            raise RuntimeError("MCP Client not connected. Call connect() first.")

        async def _call():
            try:
                result = await self.session.call_tool(
                    name=name,
                    arguments=arguments,
                    read_timeout_seconds=timedelta(seconds=self.read_timeout_seconds),
                )
                logger.info(f"MCP Tool <{name}> executed successfully")

                content = result.content
                text_content = [
                    item.text for item in content if isinstance(item, TextContent)
                ]

                if len(text_content) > 1:
                    return {"results": text_content, "count": len(text_content)}
                elif len(text_content) == 1:
                    return text_content[0]
                else:
                    return "No content returned from MCP tool"
            except Exception as e:
                logger.error(f"Error calling MCP tool {name}: {e}")
                raise

        future = asyncio.run_coroutine_threadsafe(_call(), self._loop)
        try:
            return future.result(
                timeout=self.read_timeout_seconds + 5
            )  # Add buffer for processing
        except concurrent.futures.TimeoutError:
            logger.error(f"Timeout while calling MCP tool {name}")
            raise RuntimeError(f"Timeout while calling MCP tool {name}")
        except Exception as e:
            logger.error(f"Error in call_tool for {name}: {e}")
            raise

    def list_tools(self):
        """Get a list of MCP tools synchronously."""
        if not self._connected:
            raise RuntimeError("MCP Client not connected. Call connect() first.")

        async def _list_tools():
            try:
                mcp_tools = await self.session.list_tools()
                self.tools.clear()
                for tool in mcp_tools.tools:
                    self.tools.append(MCPAgentTool(mcp_tool=tool, mcp_client=self))
                return mcp_tools
            except Exception as e:
                logger.error(f"Error listing MCP tools: {e}")
                # Clear tools on error to prevent stale state
                self.tools.clear()
                raise

        future = asyncio.run_coroutine_threadsafe(_list_tools(), self._loop)
        try:
            return future.result(timeout=self.read_timeout_seconds)
        except concurrent.futures.TimeoutError:
            logger.error("Timeout while listing MCP tools")
            raise RuntimeError("Timeout while listing MCP tools")
        except Exception as e:
            logger.error(f"Error in list_tools: {e}")
            raise

    def disconnect(self):
        """Disconnect from the MCP server synchronously."""
        if not self._connected and not self._shutdown_event:
            return

        logger.info("MCP client disconnecting...")

        # Signal shutdown to the connection task
        if self._shutdown_event:
            self._shutdown_event.set()

        # Wait for the connection task to complete (it handles cleanup via context managers)
        if self._connection_task:
            try:
                # Use a shorter timeout to avoid hanging
                self._connection_task.result(timeout=3)
                logger.info("MCP client disconnected successfully")
            except concurrent.futures.TimeoutError:
                logger.warning("MCP client disconnect timed out - forcing cleanup")
                # Cancel the task if it's still running
                if not self._connection_task.done():
                    self._connection_task.cancel()
            except Exception as e:
                logger.warning(f"Error during disconnect: {e}")
                # Try to cancel the task if there was an error
                if not self._connection_task.done():
                    self._connection_task.cancel()

        # Stop the event loop gracefully
        if self._loop and not self._loop.is_closed():
            try:
                self._loop.call_soon_threadsafe(self._loop.stop)
            except Exception as e:
                logger.warning(f"Error stopping event loop: {e}")

        # Wait for thread to finish with timeout
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
            if self._thread.is_alive():
                logger.warning("MCP client thread did not terminate within timeout")

        # Reset state
        self._connected = False
        self.session = None
        self._shutdown_event = None
        self._connection_task = None
