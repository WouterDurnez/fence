import asyncio
import concurrent.futures
import logging
import threading
from contextlib import AsyncExitStack
from datetime import timedelta
from typing import Any

from mcp.client.session import ClientSession
from mcp.types import TextContent

from fence.tools.base import BaseTool
from fence.tools.mcp.tool import MCPAgentTool
from fence.tools.mcp.utils import create_transport

logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class MCPClient:
    """Synchronous MCP Client that supports multiple transport types with a persistent event loop in a background thread."""

    def __init__(self, read_timeout_seconds: int = 30):
        self.session: ClientSession | None = None
        self.exit_stack = AsyncExitStack()
        self.tools: list[BaseTool] = []
        self.read_timeout_seconds = read_timeout_seconds
        self._connected = False
        self._loop = None
        self._thread = None
        self._executor = None

    def connect(self, transport_type: str = "stdio", **transport_kwargs):
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

        # Start the event loop in a background thread
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

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

        async def _async_connect():
            mcp_transport = create_transport(transport_type, **transport_kwargs)

            try:
                transport = await self.exit_stack.enter_async_context(mcp_transport)

                # Handle different transport return formats
                if transport_type == "stdio":
                    # stdio_client returns (read_stream, write_stream)
                    self.read_stream, self.write_stream = transport
                elif transport_type in ["sse", "streamable_http"]:
                    # SSE and streamable_http may return different formats
                    # Some return (read_stream, write_stream), others may return (read_stream, write_stream, extra)
                    if len(transport) == 2:
                        self.read_stream, self.write_stream = transport
                    else:
                        # If more than 2 values, take first 2 as read/write streams
                        self.read_stream, self.write_stream = transport[:2]
                else:
                    # Fallback for unknown transport types
                    if len(transport) >= 2:
                        self.read_stream, self.write_stream = transport[:2]
                    else:
                        raise ValueError(
                            f"Transport {transport_type} returned unexpected format: {transport}"
                        )

                self.session = await self.exit_stack.enter_async_context(
                    ClientSession(self.read_stream, self.write_stream)
                )

                await self.session.initialize()
                logger.info(f"MCP server connected using {transport_type} transport")
                self._connected = True

            except Exception as e:
                logger.error(
                    f"Error connecting to MCP server with {transport_type} transport: {e}"
                )
                await self.exit_stack.aclose()  # Clean up on connection failure
                raise e

        # Run the connection in the background loop
        future = asyncio.run_coroutine_threadsafe(_async_connect(), self._loop)
        future.result()  # Wait for completion

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

        async def _async_call():
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
                logger.error(f"Error calling MCP Tool <{name}>: {e}")
                raise e

        future = asyncio.run_coroutine_threadsafe(_async_call(), self._loop)
        return future.result()

    def list_tools(self):
        """Get a list of MCP tools synchronously."""
        if not self._connected:
            raise RuntimeError("MCP Client not connected. Call connect() first.")

        async def _async_list_tools():
            try:
                mcp_tools = await self.session.list_tools()
                self.tools.clear()

                for tool in mcp_tools.tools:
                    self.tools.append(MCPAgentTool(mcp_tool=tool, mcp_client=self))

                return mcp_tools

            except Exception as e:
                logger.error(f"Error listing MCP tools: {e}")
                raise e

        future = asyncio.run_coroutine_threadsafe(_async_list_tools(), self._loop)
        return future.result()

    def disconnect(self):
        """Disconnect from the MCP server synchronously."""
        if not self._connected:
            return

        self._connected = False
        logger.info("MCP client disconnecting...")

        if self._loop and not self._loop.is_closed():
            try:
                # Create a cleanup task that can handle the context properly
                async def _async_disconnect():
                    try:
                        # Close the exit stack which will properly clean up all contexts
                        await self.exit_stack.aclose()
                        logger.debug("AsyncExitStack closed successfully")
                    except Exception as e:
                        logger.warning(f"Error during async cleanup: {e}")
                    finally:
                        # Stop the event loop after cleanup
                        self._loop.stop()

                # Schedule the cleanup and stop in the event loop
                asyncio.run_coroutine_threadsafe(_async_disconnect(), self._loop)

                # Wait for the thread to finish with a timeout
                if self._thread and self._thread.is_alive():
                    self._thread.join(timeout=5)

                if self._thread and self._thread.is_alive():
                    logger.warning("Event loop thread did not stop gracefully")
                else:
                    logger.info("MCP client disconnected successfully")

            except Exception as e:
                logger.warning(f"Error during disconnect: {e}")

        # Clean up executor
        if self._executor:
            self._executor.shutdown(wait=False)
