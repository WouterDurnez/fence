"""
Integration tests for tool calling.
"""

import pytest

from fence.agents.agent import SuperAgent
from fence.models.openai import GPT4omini
from fence.tools.scratch import EnvTool


@pytest.mark.integration
class TestToolCalling:
    """
    Integration tests for tool calling.
    """

    def test_environment_access(self):
        """
        Test the environment access from a tool. Env variables should be passed to the tool.
        """

        # Create an agent with a model and tools
        agent = SuperAgent(
            model=GPT4omini(source="agent"),
            tools=[EnvTool()],
            environment={"some_env_var": "some_value"},
        )

        query = "Tell me what the value of the environment variable 'some_env_var' is"
        result = agent.run(query)

        assert "some_env_var" in result
        assert "some_value" in result