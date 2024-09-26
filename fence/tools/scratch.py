import json

from fence.tools.base import BaseTool


class EnvTool(BaseTool):
    """Tool to access environment variables"""

    def execute_tool(self, **kwargs) -> str:
        """Print the environment"""
        environment = kwargs.get("environment", {})
        env_vars = ", ".join([f"{k}: {v}" for k, v in environment.items()])
        return json.dumps(
            f"The environment currently holds these variables:\n) {env_vars}"
        )
