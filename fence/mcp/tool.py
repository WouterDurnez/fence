import inspect
import logging
from types import MappingProxyType
from typing import Any, Mapping

from mcp.types import Tool as MCPTool

from fence.tools.base import BaseTool, ToolParameter

logger = logging.getLogger(__name__)


class MCPAgentTool(BaseTool):

    def __init__(self, mcp_tool: MCPTool, mcp_client):
        """
        Initialize the MCPAgentTool object.

        :param mcp_tool: An instance of the MCPTool class.
        :param mcp_client: An instance of the MCPClient class.
        """
        self.mcp_tool = mcp_tool
        self.mcp_client = mcp_client

        description = mcp_tool.description or f"Tool which performs {mcp_tool.name}"

        # Build parameters using the unified ToolParameter model
        parameters = self._build_mcp_parameters(mcp_tool)

        super().__init__(description=description, parameters=parameters)

    def _build_mcp_parameters(self, mcp_tool: MCPTool) -> dict[str, ToolParameter]:
        """
        Build unified ToolParameter objects from MCP tool schema.

        :param mcp_tool: The MCP tool to extract parameters from
        :return: Dictionary of parameter name to ToolParameter objects
        """
        schema = mcp_tool.inputSchema
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))

        parameters = {}

        for name, prop in properties.items():
            # Map JSON Schema types to Python types
            json_type = prop.get("type", "string")
            type_annotation = {
                "string": str,
                "integer": int,
                "boolean": bool,
                "number": float,
                "array": list,
                "object": dict,
            }.get(json_type, str)

            # Get description from property
            description = prop.get("description")

            # Determine if required and default value
            is_required = name in required
            default_value = prop.get("default") if not is_required else None

            parameters[name] = ToolParameter(
                name=name,
                type_annotation=type_annotation,
                description=description,
                required=is_required,
                default_value=default_value,
            )

        return parameters

    def _build_unified_parameters(
        self, explicit_parameters: dict[str, ToolParameter] | None = None
    ) -> dict[str, ToolParameter]:
        """Build unified parameters for MCP tool, avoiding circular dependency."""
        if explicit_parameters:
            return explicit_parameters

        # Use the MCP-specific parameter building method
        return self._build_mcp_parameters(self.mcp_tool)

    def get_tool_name(self) -> str:
        """
        Get the name of the tool.

        :return: The name of the MCP tool
        """
        return self.mcp_tool.name

    def get_tool_params(self) -> Mapping[str, inspect.Parameter]:
        """
        Get the parameters of the tool as a mapping of parameter names to Parameter objects.

        This method creates inspect.Parameter objects from the unified parameters.
        A **kwargs parameter is always added to support additional arguments.

        :return: A read-only mapping of parameter names to Parameter objects
        """
        parameters = []

        # Build parameters from unified parameter model
        for param_name, param in self.parameters.items():
            if param.required:
                param_obj = inspect.Parameter(
                    param_name,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=param.type_annotation,
                )
            else:
                param_obj = inspect.Parameter(
                    param_name,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=param.type_annotation | None,
                    default=param.default_value,
                )
            parameters.append(param_obj)

        # Add **kwargs
        parameters.append(inspect.Parameter("kwargs", inspect.Parameter.VAR_KEYWORD))
        logger.info(f"Parameters: {parameters}")

        return MappingProxyType({p.name: p for p in parameters})

    def execute_tool(self, environment: dict[str, Any] | None = None, **kwargs) -> Any:
        """Execute the MCP tool with the given arguments.

        :param environment: Optional dictionary containing environment variables
        :param kwargs: Additional keyword arguments to pass to the tool
        :return: The result of the tool execution
        """
        logger.info(
            f"Executing MCP tool {self.get_tool_name()} with arguments: {kwargs}"
        )

        # Remove environment from kwargs if it exists to avoid passing it to MCP tool
        filtered_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k != "environment" and k in self.parameters
        }

        try:
            response = self.mcp_client.call_tool(
                name=self.get_tool_name(), arguments=filtered_kwargs
            )
            return response
        except Exception as e:
            return {
                "error": f"Failed to execute MCP tool '{self.get_tool_name()}': {str(e)}"
            }
