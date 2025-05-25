import inspect
import logging
import os
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional

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
    
    def get_tool_signature(self) -> Mapping[str, inspect.Parameter]:
        """
        Get the tool signature as a mapping of parameter names to Parameter objects.
        
        This method parses the inputSchema of the MCP tool and converts it to a set of
        inspect.Parameter objects that define the signature of the tool function.
        
        It maps JSON Schema types to Python types:
        - string -> str
        - integer -> int
        - boolean -> bool
        - number -> float
        - others -> str (default)
        
        Required parameters are included without default values.
        Optional parameters are included with None as default value.
        A **kwargs parameter is always added to support additional arguments.

        Returns:
            Mapping[str, inspect.Parameter]: A read-only mapping of parameter names to Parameter objects
        """
        schema = self.mcp_tool.inputSchema
        parameters = []

        properties = schema.get("properties", {})
        required = set(schema.get("required", []))

        for name, prop in properties.items():
            # Map JSON Schema types to Python types
            json_type = prop.get("type", "string")
            annotation = {
                "string": str,
                "integer": int,
                "boolean": bool,
                "number": float
            }.get(json_type, str)

            if name in required:
                param = inspect.Parameter(
                    name,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=annotation
                )
            else:
                default_value = prop.get("default", None)
                param = inspect.Parameter(
                    name,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=annotation,
                    default=default_value
                )

            parameters.append(param)

        # Add **kwargs
        parameters.append(inspect.Parameter(
            "kwargs",
            inspect.Parameter.VAR_KEYWORD
        ))

        return MappingProxyType({p.name: p for p in parameters})
    
    def get_tool_params(self) -> Mapping[str, inspect.Parameter]:
        """
        Get the parameters of the tool.

        This is a convenience method that calls get_tool_signature().
        
        Returns:
            Mapping[str, inspect.Parameter]: A read-only mapping of parameter names to Parameter objects
        """
        return self.get_tool_signature()
    


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

        return response
