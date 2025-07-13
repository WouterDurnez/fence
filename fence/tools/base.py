"""
Tools for agents
"""

import inspect
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, get_args

from pydantic import BaseModel, ConfigDict, Field

from fence.utils.docstring_parser import DocstringParser

# Union: For type hints when we need to reference Union types
# get_origin: Gets the origin of a type (e.g., Union, List, etc.)
# get_args: Gets the arguments of a type (e.g., for Union[int, None], returns (int, type(None)))


logger = logging.getLogger(__name__)

####################
# Parameter Models #
####################


class ToolParameter(BaseModel):
    """A unified model for tool parameters combining type info and descriptions.

    This class represents a single parameter for a tool, including its type information,
    description, whether it's required, and any default value. It handles both simple
    types (like `int`, `str`) and complex types (like `int | None`, `list[str]`).
    """

    name: str = Field(..., description="Parameter name")
    type_annotation: Any = Field(..., description="Parameter type annotation")
    description: str | None = Field(None, description="Parameter description")
    required: bool = Field(..., description="Whether the parameter is required")
    default_value: Any = Field(
        None, description="Default value if parameter is optional"
    )

    # Allow arbitrary types in Pydantic validation to handle complex type annotations
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def json_type(self) -> str:
        """Convert Python type to JSON schema type string.

        This method takes a Python type annotation (which could be a simple type like `int`
        or a complex type like `int | None`) and converts it to a JSON schema type string
        that can be used in API specifications.

        Examples:
            - `int` -> "integer"
            - `str` -> "string"
            - `int | None` -> "integer" (takes the first non-None type)
            - `list[str]` -> "array"
            - `dict` -> "object"
            - Unknown types -> "string" (fallback)

        :return: The JSON schema type string
        """

        # Check if this is a union type (like 'int | None', 'str | int', etc.)
        # Union types have a '__args__' attribute that contains the individual types
        if hasattr(self.type_annotation, "__args__"):

            # Extract the individual types from the union using get_args()
            # For 'int | None', this would return (int, type(None))
            args = get_args(self.type_annotation)

            # For union types, we want to find the first non-None type
            # This is useful for cases like 'int | None' where we want 'int'
            # We iterate through the args and take the first one that isn't None
            for arg in args:
                if arg is not type(None):  # Skip None types
                    type_name = getattr(arg, "__name__", "string")
                    break
            else:
                # If all args are None (unlikely but possible), fallback to string
                type_name = "string"
        else:
            # This is a simple type (like 'int', 'str', etc.)
            # Get the type name, with 'string' as fallback if __name__ doesn't exist
            type_name = getattr(self.type_annotation, "__name__", "string")

        # Map Python type names to JSON schema type strings
        # This mapping is used by various LLM APIs and tools that expect JSON schema
        # The mapping follows standard JSON schema conventions
        type_mapping = {
            "str": "string",
            "int": "integer",
            "float": "number",
            "bool": "boolean",
            "list": "array",
            "dict": "object",
        }

        # Return the mapped type, or "string" as fallback for unknown types
        # This ensures we always return a valid JSON schema type
        return type_mapping.get(type_name, "string")


##############
# Base class #
##############


class BaseTool(ABC):

    def __init__(
        self,
        description: str | None = None,
        parameters: dict[str, ToolParameter] | None = None,
    ):
        """
        Initialize the BaseTool object.

        :param description: A description of the tool (if not provided, the docstring will be used)
        :param parameters: Dict of ToolParameter objects (if not provided, will be extracted from execute_tool signature)
        """
        self.description = description
        self.environment = {}

        # Build unified parameters from inspect signature and descriptions
        self.parameters = parameters or self._build_unified_parameters()

    def _build_unified_parameters(
        self, explicit_parameters: dict[str, ToolParameter] | None = None
    ) -> dict[str, ToolParameter]:
        """Build unified parameters combining signature inspection and descriptions."""
        if explicit_parameters:
            return explicit_parameters

        # Get signature parameters
        signature_params = self.get_tool_params()

        # Get parameter descriptions from docstring
        descriptions = self._get_docstring_param_descriptions()

        # Build unified parameters
        unified_params = {}
        for param_name, param in signature_params.items():
            if param_name in ["environment", "kwargs"]:
                continue

            # Get type annotation
            type_annotation = (
                param.annotation if param.annotation != inspect.Parameter.empty else str
            )

            # Check if required
            required = param.default == inspect.Parameter.empty
            default_value = param.default if not required else None

            # Get description
            description = descriptions.get(param_name)

            unified_params[param_name] = ToolParameter(
                name=param_name,
                type_annotation=type_annotation,
                description=description,
                required=required,
                default_value=default_value,
            )

        return unified_params

    def run(self, environment: dict = None, **kwargs):
        """
        Base method that is called when the tool is used.
        This method will always accept 'environment' as a parameter.
        """

        # Initialize environment if not provided
        self.environment = environment or self.environment

        # Pass the environment to the execute_tool method
        kwargs["environment"] = environment

        # Call the subclass-specific implementation
        return self.execute_tool(**kwargs)

    @abstractmethod
    def execute_tool(self, environment: dict = None, **kwargs):
        """
        The method that should be implemented in the derived class.
        This will handle the actual tool-specific logic.

        :param environment: A dictionary of environment variables to pass to the tool
        :param kwargs: Additional keyword arguments for the tool
        """
        raise NotImplementedError

    def get_tool_name(self):
        """
        Get the name of the tool.
        """
        return self.__class__.__name__

    def get_tool_description(self):
        """
        Get the description of the tool.
        """
        return self.description or self.__doc__ or "No description provided"

    def get_tool_params(self):
        """
        Get the raw parameters of the tool from signature inspection.
        """
        return inspect.signature(self.execute_tool).parameters

    def _get_docstring_param_descriptions(self) -> dict[str, str]:
        """
        Get parameter descriptions from execute_tool method's docstring.

        :return: A dict of param name -> description, None if no description found.
        """
        # Get all parameters to ensure complete coverage
        params = self.get_tool_params()
        result = {param: None for param in params}

        # Try to get descriptions from docstring parsing
        parser = DocstringParser()
        docstring_descriptions = parser.parse(self.execute_tool)
        result.update(docstring_descriptions)

        return result

    def get_representation(self):
        """
        Get the representation of the tool.
        """
        tool_name = self.get_tool_name()
        tool_description = self.get_tool_description()

        # Format parameters using unified parameters
        formatted_params = []
        for param_name, param in self.parameters.items():
            # Get parameter type
            param_type = param.type_annotation.__name__

            # Check if parameter is required
            required_str = "(required)" if param.required else "(optional)"

            # Get parameter description if available
            desc_str = f" - {param.description}" if param.description else ""

            # Add formatted parameter
            formatted_params.append(
                f"{param_name}: {param_type} {required_str}{desc_str}"
            )

        # Join parameters or show "None" if no parameters
        params_str = (
            "\n  - " + "\n  - ".join(formatted_params) if formatted_params else "None"
        )

        return f"""### {tool_name}
- Description: {tool_description}
- Parameters: {params_str}
"""

    def format_toml(self):
        """
        Returns a TOML-formatted key-value pair of the tool name,
        the description (docstring) of the tool, and the arguments
        of the `run` method.
        """
        # Preformat the arguments using unified parameters
        argument_string = ""
        if self.parameters:
            for param_name, param in self.parameters.items():
                argument_string += (
                    f"[[tools.tool_params]]\n"
                    f'name = "{param_name}"\n'
                    f'type = "{param.type_annotation.__name__}"\n'
                    f'description = "{param.description}"\n'
                )
        else:
            argument_string = "# No arguments"

        # Get the description of the tool, if not provided, use the docstring
        tool_description = self.description or self.__doc__

        if not tool_description:
            logger.warning(
                f"Tool {self.__class__.__name__} has no description or docstring."
            )

        # Format the TOML representation of the tool for the agent
        toml_string = f"""[[tools]]
tool_name = "{self.__class__.__name__}"
tool_description = "{tool_description}"
{argument_string}"""

        return toml_string

    def model_dump_bedrock_converse(self):
        """
        Dump the tool in the format required by Bedrock Converse.

        Example output:
        ```
        {
            "toolSpec": {
                "name": "top_song",
                "description": "Get the most popular song played on a radio station.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "sign": {
                                "type": "string",
                                "description": "The call sign for the radio station for which you want the most popular song. Example calls signs are WZPZ, and WKRP.",
                            }
                        },
                        "required": ["sign"],
                    }
                }
            }
        }
        ```
        """
        properties = {}
        required = []

        # Use unified parameters
        for param_name, param in self.parameters.items():
            # Use actual parameter description if available, otherwise fallback to generic description
            description = (
                param.description
                or f"Parameter {param_name} for the {self.__class__.__name__} tool"
            )

            properties[param_name] = {
                "type": param.json_type,
                "description": description,
            }

            if param.required:
                required.append(param_name)

        return {
            "toolSpec": {
                "name": self.get_tool_name(),
                "description": self.get_tool_description(),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    }
                },
            }
        }


##############
# Decorators #
##############


def tool(func_or_description=None, *, description=None):
    """
    A decorator to turn a function into a tool that can be executed with the BaseTool interface.

    Can be used as:
    - @tool (uses function's docstring)
    - @tool() (uses function's docstring)
    - @tool("custom description") (uses provided description as positional arg)
    - @tool(description="custom description") (uses provided description as keyword arg)

    :param func_or_description: Either a function (when used without parentheses) or a description string
    :param description: Description string as keyword argument
    """

    def decorator(func: Callable, tool_description: str = None):
        # Dynamically create the class with the capitalized function name
        class_name = "".join(
            [element.capitalize() for element in func.__name__.split("_")]
        )
        class_name = (
            f"{class_name}Tool" if not class_name.endswith("Tool") else class_name
        )

        # Get the original function's signature to see which params it accepts
        func_signature = inspect.signature(func)
        func_params = set(func_signature.parameters.keys())

        # Define the execute_tool method with proper handling of environment
        def execute_tool_wrapper(self, **kwargs):
            # Filter kwargs to only include parameters that the function accepts
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in func_params}
            return func(**filtered_kwargs)

        # Custom get_tool_params method for the decorated function
        def get_tool_params(self):
            return func_signature.parameters

        # Custom _get_docstring_param_descriptions method for the decorated function
        def _get_docstring_param_descriptions_override(self) -> dict[str, str]:
            """Get parameter descriptions from decorated function's docstring."""
            from fence.utils.docstring_parser import DocstringParser

            # Get all parameters to ensure complete coverage
            params = func_signature.parameters
            result = {param: None for param in params}

            # Try to get descriptions from docstring parsing
            parser = DocstringParser()
            docstring_descriptions = parser.parse(func)
            result.update(docstring_descriptions)

            return result

        # Custom _build_unified_parameters method for the decorated function
        def _build_unified_parameters_override(
            self, explicit_parameters: dict[str, ToolParameter] | None = None
        ) -> dict[str, ToolParameter]:
            """Build unified parameters for decorated function."""
            if explicit_parameters:
                return explicit_parameters

            # Get signature parameters
            signature_params = func_signature.parameters

            # Get parameter descriptions using the custom method
            descriptions = self._get_docstring_param_descriptions()

            # Build unified parameters
            unified_params = {}
            for param_name, param in signature_params.items():
                if param_name in ["environment", "kwargs"]:
                    continue

                # Get type annotation
                type_annotation = (
                    param.annotation
                    if param.annotation != inspect.Parameter.empty
                    else str
                )

                # Check if required
                required = param.default == inspect.Parameter.empty
                default_value = param.default if not required else None

                # Get description
                description = descriptions.get(param_name)

                unified_params[param_name] = ToolParameter(
                    name=param_name,
                    type_annotation=type_annotation,
                    description=description,
                    required=required,
                    default_value=default_value,
                )

            return unified_params

        # Define the class dynamically
        ToolClass = type(
            class_name,
            (BaseTool,),
            {
                "__init__": lambda self: BaseTool.__init__(
                    self, description=tool_description or func.__doc__
                ),
                "execute_tool": execute_tool_wrapper,
                "get_tool_params": get_tool_params,
                "_get_docstring_param_descriptions": _get_docstring_param_descriptions_override,
                "_build_unified_parameters": _build_unified_parameters_override,
            },
        )

        return ToolClass()

    # Handle different usage patterns
    if func_or_description is None:
        # @tool() or @tool(description="...") - return decorator
        return lambda func: decorator(func, description)
    elif isinstance(func_or_description, str):
        # @tool("description") - return decorator that uses provided description
        return lambda func: decorator(func, func_or_description)
    elif callable(func_or_description):
        # @tool - direct decoration, use docstring
        if description is not None:
            raise ValueError(
                "Cannot use both positional function and keyword description"
            )
        return decorator(func_or_description, None)
    else:
        raise ValueError("Invalid usage of @tool decorator")


if __name__ == "__main__":

    # Test different usage patterns

    # 1. @tool with positional description
    @tool("A tool that returns the current time")
    def get_current_time(location: str):
        """Get current time for a location."""
        return f"The current time in {location} is 12:00 PM"

    # 2. @tool() without description (uses docstring)
    @tool()
    def get_weather(city: str):
        """Get weather information for a city."""
        return f"The weather in {city} is sunny"

    # 3. @tool without parentheses (uses docstring)
    @tool
    def calculate_sum(a: int, b: int):
        """Calculate the sum of two numbers."""
        return a + b

    # 4. @tool with keyword description
    @tool(description="A tool that converts temperature units")
    def convert_temperature(
        temp: float, from_unit: str = "celsius", to_unit: str = "fahrenheit"
    ):
        """Convert temperature between units.

        :param temp: Temperature value to convert
        :param from_unit: Source temperature unit
        :param to_unit: Target temperature unit
        :return: Converted temperature
        """
        if from_unit == "celsius" and to_unit == "fahrenheit":
            return (temp * 9 / 5) + 32
        elif from_unit == "fahrenheit" and to_unit == "celsius":
            return (temp - 32) * 5 / 9
        else:
            return temp  # Same unit or unsupported conversion

    print("=== Tool with positional description ===")
    print(get_current_time.get_tool_description())
    print(get_current_time.run(location="New York"))

    print("\n=== Tool with empty parentheses (docstring) ===")
    print(get_weather.get_tool_description())
    print(get_weather.run(city="Paris"))

    print("\n=== Tool without parentheses (docstring) ===")
    print(calculate_sum.get_tool_description())
    print(calculate_sum.run(a=5, b=3))

    print("\n=== Tool with keyword description ===")
    print(convert_temperature.get_tool_description())
    print(convert_temperature.run(temp=25.0, from_unit="celsius", to_unit="fahrenheit"))
