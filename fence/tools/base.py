"""
Tools for agents
"""

import inspect
import logging
from abc import ABC, abstractmethod
from typing import Callable

logger = logging.getLogger(__name__)

##############
# Base class #
##############


class BaseTool(ABC):

    def __init__(self, description: str = None):
        """
        Initialize the BaseTool object.

        :param description: A description of the tool (if not provided, the docstring will be used)
        """
        self.description = description
        self.environment = {}

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
        Get the parameters of the tool.
        """
        return inspect.signature(self.execute_tool).parameters

    def get_representation(self):
        """
        Get the representation of the tool.
        """
        tool_name = self.get_tool_name()
        tool_description = self.get_tool_description()
        tool_params = self.get_tool_params()

        # Format parameters in a more readable way
        formatted_params = []
        for name, param in tool_params.items():
            if name in ["environment", "kwargs"]:
                continue

            # Get parameter type
            param_type = (
                param.annotation.__name__
                if param.annotation != inspect.Parameter.empty
                else "str"
            )

            # Check if parameter is required
            is_optional = param.default != inspect.Parameter.empty
            required_str = "(optional)" if is_optional else "(required)"

            # Add formatted parameter
            formatted_params.append(f"{name}: {param_type} {required_str}")

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
        # Get the arguments of the run method
        run_args = self.get_tool_params()

        # Add all arguments, and ensure that the argument
        # is annotated with str if no type is provided
        run_args = {
            arg_name: (
                arg_type.annotation
                if arg_type.annotation != inspect.Parameter.empty
                else str
            )
            for arg_name, arg_type in run_args.items()
        }
        run_args.pop("environment", None)
        run_args.pop("kwargs", None)

        # Preformat the arguments
        argument_string = ""
        if run_args:
            for arg_name, arg_type in run_args.items():
                argument_string += (
                    f"[[tools.tool_params]]\n"
                    f'name = "{arg_name}"\n'
                    f'type = "{arg_type.__name__ or str}"\n'
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
        # Get the tool's parameters from its execute_tool method
        run_args = self.get_tool_params()
        properties = {}
        required = []

        for arg_name, arg_type in run_args.items():
            if arg_name in ["environment", "kwargs"]:
                continue

            # Get the type annotation or default to string
            type_annotation = (
                arg_type.annotation
                if arg_type.annotation != inspect.Parameter.empty
                else str
            )
            type_name = (
                type_annotation.__name__
                if hasattr(type_annotation, "__name__")
                else "string"
            )

            # Convert Python types to JSON schema types
            json_type = {
                "str": "string",
                "int": "integer",
                "float": "number",
                "bool": "boolean",
                "list": "array",
                "dict": "object",
            }.get(type_name, "string")

            properties[arg_name] = {
                "type": json_type,
                "description": f"Parameter {arg_name} for the {self.__class__.__name__} tool",
            }

            if arg_type.default == inspect.Parameter.empty:
                required.append(arg_name)

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


def tool(func_or_description=None):
    """
    A decorator to turn a function into a tool that can be executed with the BaseTool interface.

    Can be used as:
    - @tool (uses function's docstring)
    - @tool() (uses function's docstring)
    - @tool("custom description") (uses provided description)

    :param func_or_description: Either a function (when used without parentheses) or a description string
    """

    def decorator(func: Callable, description: str = None):
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

        # Define the class dynamically
        ToolClass = type(
            class_name,
            (BaseTool,),
            {
                "__init__": lambda self: BaseTool.__init__(
                    self, description=description or func.__doc__
                ),
                "execute_tool": execute_tool_wrapper,
                "get_tool_params": get_tool_params,
            },
        )

        return ToolClass()

    # Handle different usage patterns
    if func_or_description is None:
        # @tool() - return decorator that uses docstring
        return lambda func: decorator(func, None)
    elif isinstance(func_or_description, str):
        # @tool("description") - return decorator that uses provided description
        return lambda func: decorator(func, func_or_description)
    elif callable(func_or_description):
        # @tool - direct decoration, use docstring
        return decorator(func_or_description, None)
    else:
        raise ValueError("Invalid usage of @tool decorator")


if __name__ == "__main__":

    # Test different usage patterns

    # 1. @tool with description
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

    print("=== Tool with description ===")
    print(get_current_time.get_tool_description())
    print(get_current_time.run(location="New York"))

    print("\n=== Tool with empty parentheses (docstring) ===")
    print(get_weather.get_tool_description())
    print(get_weather.run(city="Paris"))

    print("\n=== Tool without parentheses (docstring) ===")
    print(calculate_sum.get_tool_description())
    print(calculate_sum.run(a=5, b=3))
