from fence.tools.base import BaseTool


class EnvTool(BaseTool):
    """Tool to access environment variables"""

    def execute_tool(self, environment: dict = None, **kwargs) -> str:
        """
        Print the environment variables.

        :param environment: Dictionary containing environment variables
        :param kwargs: Additional parameters (not used)
        :return: A formatted string showing environment variables
        """
        # Use the environment passed as parameter, or empty dict if None
        environment = environment or {}

        # Format each variable for display
        if environment:
            env_vars = "\n".join([f"{k}: {v}" for k, v in environment.items()])
            return f"The environment currently holds these variables:\n{env_vars}"
        else:
            return "The environment is empty or not accessible."
