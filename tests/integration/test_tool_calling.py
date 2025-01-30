"""
Integration tests for tool calling.
"""

from fence.agents.agent import Agent
from fence.models.openai.gpt import GPT4omini
from fence.tools.scratch import EnvTool


class TestToolCalling:
    """
    Integration tests for tool calling.
    """

    def test_environment_access(self):
        """
        Test the environment access from a tool. Env variables should be passed to the tool.
        """

        # Create an agent with a model and tools
        agent = Agent(
            model=GPT4omini(source="agent"),
            tools=[EnvTool()],
            environment={"some_env_var": "some_value"},
        )

        query = "Tell me what the value of the environment variable 'some_env_var' is"
        result = agent.run(query)

        assert "some_env_var" in result
        assert "some_value" in result
