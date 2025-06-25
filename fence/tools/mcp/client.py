import logging
from contextlib import AsyncExitStack

from mcp import StdioServerParameters
from mcp.client import ClientSession

from fence.tools.base import BaseTool

logger = logging.getLogger(__name__)


class MCPClient:

    def __init__(self):
        self.session: ClientSession | None = None
        self.exit_stack = AsyncExitStack()
        self.tools: list[BaseTool] = []

    # Connect to the MCP server
    async def connect(self, server_script_path: str):

        try:

            server_parameters = StdioServerParameters()
            self.session = ClientSession(server_parameters)

            await self.session.initialize()

            self.model = self.session.get_model("llama3.1")

        except Exception as e:
            logger.error(f"Error connecting to MCP server: {e}")
            raise e

    # Call an MCP tool

    # Get MCP tools

    # Process query

    # Cleanup
