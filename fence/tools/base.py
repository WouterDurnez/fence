"""
Tools for agents
"""

from abc import ABC


class BaseTool(ABC):

    def run(self, **kwargs):
        """
        The method that will be called when the tool is used.
        The method should be implemented in the derived class.
        """
        raise NotImplementedError

    def format_toml(self):
        """
        Returns a TOML-formatted key-value pair of the tool name,
        the description (docstring) of the tool, and the arguments
        of the `run` method.
        """

        # Get the arguments of the run method
        run_args = self.run.__annotations__
        run_args.pop("return", None)

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
