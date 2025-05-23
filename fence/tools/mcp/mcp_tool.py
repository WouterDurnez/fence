import logging
import os
from typing import TYPE_CHECKING, Any, Dict, Optional

from mcp.types import Tool as MCPTool

from fence.tools.base import BaseTool

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .mcp_client import MCPClient

class MCPAgentTool(BaseTool):

    def __init__(self, mcp_tool: MCPTool, mcp_client: "MCPClient"):
        """
        Initialize the MCPAgentTool object.

        :param mcp_tool: An instance of the MCPTool class.
        :param mcp_client: An instance of the MCPClient class.
        """
        description = mcp_tool.description or f"Tool which performs {mcp_tool.name}"
        super().__init__(description=description)
        self.mcp_tool = mcp_tool
        self.mcp_client = mcp_client

    def get_tool_name(self) -> str:
        """
        Get the name of the tool.

        :return: The name of the MCP tool
        """
        return self.mcp_tool.name

    def model_dump_bedrock_converse(self) -> Dict[str, Any]:
        """
        Dump the tool in the format required by Bedrock Converse.

        :return: A dictionary containing the tool specification in Bedrock Converse format
        """
        return {
            "toolSpec": {
                "name": self.get_tool_name(),
                "description": self.get_tool_description(),
                "inputSchema": {
                    "json": self.mcp_tool.inputSchema
                }
            }
        }
    
    def execute_tool(self, environment: Optional[Dict[str, Any]] = None, **kwargs) -> Any:
        """
        Execute the MCP tool with the given arguments.

        :param environment: Optional dictionary containing environment variables
        :param kwargs: Additional keyword arguments to pass to the tool
        :return: The result of the tool execution
        """
        logger.info(f"Executing MCP tool {self.get_tool_name()} with arguments: {kwargs}")

        # Generate a unique tool use ID if not provided
        tool_use_id = kwargs.pop("tool_use_id", f"mcp_{self.get_tool_name()}_{os.urandom(4).hex()}")

        response =  self.mcp_client.call_tool_sync(
            tool_use_id=tool_use_id,
            name=self.get_tool_name(),
            arguments=kwargs
        )

        print(response)

        return response

