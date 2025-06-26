import asyncio
import logging
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession

from fence.tools.base import BaseTool
from fence.tools.mcp.utils import create_stdio_transport

logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class MCPClient:

    def __init__(self):
        self.session: ClientSession | None = None
        self.exit_stack = AsyncExitStack()
        self.tools: list[BaseTool] = []

    # Connect to the MCP server
    async def connect(self, mcp_transport: Any):
        try:
            # mcp_transport is already an async context manager, not a callable
            stdio_transport = await self.exit_stack.enter_async_context(mcp_transport)
            self.stdio, self.write = stdio_transport

            self.session = await self.exit_stack.enter_async_context(
                ClientSession(self.stdio, self.write)
            )

            await self.session.initialize()
            logger.info("MCP server connected")

            mcp_tools = await self.session.list_tools()
            logger.info(f"MCP tools: {mcp_tools.tools}")
            for tool in mcp_tools.tools:
                logger.info(
                    f"\n\nTool: {tool.name}, Description: {tool.description}, Params: {tool.inputSchema}"
                )
        except Exception as e:
            logger.error(f"Error connecting to MCP server: {e}")
            raise e

    # Call an MCP tool

    # Get MCP tools
    async def list_tools(self):
        try:
            mcp_tools = await self.session.list_tools()

            # Convert MCP tools to a list of BaseTool
            for tool in mcp_tools.tools:
                self.tools.append(
                    BaseTool(
                        name=tool.name,
                        description=tool.description,
                        params=tool.input_schema.model_dump(),
                    )
                )

            return mcp_tools

        except Exception as e:
            logger.error(f"Error listing MCP tools: {e}")
            raise e

    # Process query

    # Cleanup
    async def disconnect(self):
        """Properly close all async context managers"""
        try:
            await self.exit_stack.aclose()
            logger.info("MCP client disconnected")
        except Exception as e:
            logger.error(f"Error disconnecting from MCP server: {e}")
            raise e


async def main():
    """Main function that properly handles connection and cleanup"""
    command = "uvx"
    args = ["awslabs.aws-documentation-mcp-server@latest"]

    mcp_transport = create_stdio_transport(command, args)

    client = MCPClient()
    await client.connect(mcp_transport)
    await client.disconnect()

    print(client.tools)


if __name__ == "__main__":
    asyncio.run(main())
