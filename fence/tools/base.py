"""
Tools for agents
"""

from abc import ABC, abstractmethod


class BaseTool(ABC):

    def run(self, environment: dict = None, **kwargs):
        """
        Base method that is called when the tool is used.
        This method will always accept 'environment' as a parameter.
        """

        # Initialize environment if not provided
        environment = environment or {}

        # Preprocess environment if needed
        self.preprocess_environment(environment)

        # Pass the environment to the execute_tool method
        kwargs["environment"] = environment

        # Call the subclass-specific implementation
        return self.execute_tool(**kwargs)

    @abstractmethod
    def execute_tool(self, **kwargs):
        """
        The method that should be implemented in the derived class.
        This will handle the actual tool-specific logic.
        """
        raise NotImplementedError

    def preprocess_environment(self, environment: dict):
        """
        Optional method to preprocess the environment, can be overridden by subclasses.
        """
        pass

    def format_toml(self):
        """
        Returns a TOML-formatted key-value pair of the tool name,
        the description (docstring) of the tool, and the arguments
        of the `run` method.
        """
        # Get the arguments of the run method
        run_args = self.execute_tool.__annotations__
        run_args.pop("return", None)
        run_args.pop("environment", None)

        # Preformat the arguments
        argument_string = ""
        if run_args:
            for arg_name, arg_type in run_args.items():
                argument_string += (
                    f"[[tools.tool_params]]\n"
                    f'name = "{arg_name}"\n'
                    f'type = "{arg_type.__name__}"\n'
                )
        else:
            argument_string = "# No arguments"

        toml_string = f"""[[tools]]
tool_name = "{self.__class__.__name__}"
tool_description = "{self.__doc__}"
{argument_string}"""

        return toml_string
