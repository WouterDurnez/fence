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
from fence.tools.mcp.utils import create_stdio_transport

logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class MCPClient:
    """Synchronous MCP Client that uses a persistent event loop in a background thread."""

    def __init__(self, read_timeout_seconds: int = 30):
        self.session: ClientSession | None = None
        self.exit_stack = AsyncExitStack()
        self.tools: list[BaseTool] = []
        self.read_timeout_seconds = read_timeout_seconds
        self._connected = False
        self._loop = None
        self._thread = None
        self._executor = None

    def connect(self, command: str, args: list[str]):
        """Connect to the MCP server synchronously."""

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
            mcp_transport = create_stdio_transport(command, args)

            try:
                stdio_transport = await self.exit_stack.enter_async_context(
                    mcp_transport
                )
                self.stdio, self.write = stdio_transport

                self.session = await self.exit_stack.enter_async_context(
                    ClientSession(self.stdio, self.write)
                )

                await self.session.initialize()
                logger.info("MCP server connected")
                self._connected = True

            except Exception as e:
                logger.error(f"Error connecting to MCP server: {e}")
                raise e

        # Run the connection in the background loop
        future = asyncio.run_coroutine_threadsafe(_async_connect(), self._loop)
        future.result()  # Wait for completion

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

        # Stop the event loop gracefully - this will trigger cleanup
        if self._loop and not self._loop.is_closed():
            try:
                # Schedule a stop on the event loop
                self._loop.call_soon_threadsafe(self._loop.stop)

                # Wait for the thread to finish with a timeout
                if self._thread and self._thread.is_alive():
                    self._thread.join(timeout=3)

                if self._thread and self._thread.is_alive():
                    logger.warning("Event loop thread did not stop gracefully")
                else:
                    logger.info("MCP client disconnected successfully")

            except Exception as e:
                logger.warning(f"Error during disconnect: {e}")

        # Clean up executor
        if self._executor:
            self._executor.shutdown(wait=False)
