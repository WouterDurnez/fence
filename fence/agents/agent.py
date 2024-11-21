"""
Agent class to orchestrate a flow that potentially calls tools or other, specialized agents.
"""

import logging

from fence import LLM, ClaudeInstant, Link, TOMLParser, setup_logging
from fence.agents.base import BaseAgent
from fence.links import logger as link_logger
from fence.memory.base import BaseMemory
from fence.prompts.agents import REACT_MULTI_AGENT_TOOL_PROMPT
from fence.tools.base import BaseTool
from fence.tools.math import CalculatorTool, PrimeTool
from fence.tools.scratch import EnvTool
from fence.tools.text import TextInverterTool

logger = logging.getLogger(__name__)

# Suppress the link logger
link_logger.setLevel("WARNING")


class Agent(BaseAgent):
    """An LLM-based agent capable of delegating tasks to other agents or tools."""

    def __init__(
        self,
        identifier: str | None = None,
        model: LLM | None = None,
        description: str | None = None,
        role: str | None = None,
        delegates: list[BaseAgent] | None = None,
        tools: list[BaseTool] | None = None,
        memory: BaseMemory | None = None,
        environment: dict | None = None,
    ):
        """
        Initialize the Agent object.

        :param identifier: An identifier for the agent. If none is provided, the class name will be used.
        :param model: An LLM model object.
        :param description: A description of the agent. Used to represent the agent to other agents.
        :param role: The role of the agent. Used to represent the agent to itself, detailing its purpose.
        :param delegates: A list of delegate agents.
        :param tools: A list of Tool objects.
        :param memory: A memory class. Defaults to FleetingMemory.
        :param environment: A dictionary of environment variables to pass to delegates and tools.
        """
        super().__init__(
            identifier=identifier,
            model=model,
            description=description,
            memory=memory,
            environment=environment or {},
        )

        # Set role
        self.role = (
            role
            or "You are a general purpose agent, capable of delegating tasks to other agents or tools."
        )

        # Delegates and tools initialization
        self.delegates = (
            {delegate.identifier: delegate for delegate in delegates}
            if delegates
            else {}
        )
        # Add environment variables to delegates
        for delegate in self.delegates.values():
            delegate.environment.update(self.environment)

        # Initialize tools with environment variables
        if tools:
            for tool in tools:
                tool.environment.update(self.environment)
            self.tools = {tool.__class__.__name__: tool for tool in tools}
        else:
            self.tools = {}

        # Logging agent creation details
        logger.info(
            f"Initialized agent with model <{model.model_name}>, delegates: {list(self.delegates.keys())}, tools: {list(self.tools.keys())}"
        )

        # Prepare formatted delegates and tools
        self.formatted_delegates = self._format_entities(list(self.delegates.values()))
        self.formatted_tools = self._format_entities(list(self.tools.values()))

        # Set system message
        self._system_message = REACT_MULTI_AGENT_TOOL_PROMPT.format(
            role=self.role,
            delegates=self.formatted_delegates,
            tools=self.formatted_tools,
        )

        # Initialize the memory buffer
        self._flush_memory()

    def run(self, prompt: str) -> str:
        """
        Main execution loop to process the input prompt.

        :param prompt: The input prompt to process.
        :return: The final answer from the agent.
        """

        # Add initial prompt to the memory buffer
        self.memory.add_message(role="user", content=prompt)

        max_iterations, iteration_count = 30, 0
        response, answer = None, None

        while iteration_count < max_iterations:

            link = Link(
                name=f"{self.identifier or 'agent'}_step_{iteration_count}",
                model=self.model,
                template=self.memory.to_messages_template(),
            )
            response = link.run()["state"]
            logger.info(f"Model response: {response}")

            self.memory.add_message(role="assistant", content=response)

            if "[ANSWER]" in response:
                answer = self._extract_answer(response)
                break

            if "[DELEGATE]" in response:
                delegate_response = self._handle_delegate_action(response)
                if delegate_response:
                    self.memory.add_message(
                        role="user", content=f"[OBSERVATION] {delegate_response}"
                    )

            if "[ACTION]" in response:
                tool_response = self._handle_tool_action(response)
                if tool_response:
                    self.memory.add_message(
                        role="user", content=f"[OBSERVATION] {tool_response}"
                    )

            iteration_count += 1

        self._flush_memory()
        return answer or "No answer found"

    @staticmethod
    def _format_entities(entities: list) -> str:
        """Format delegates or tools into TOML representation."""
        return (
            "".join(entity.format_toml() for entity in entities)
            if entities
            else "None available"
        )

    @staticmethod
    def _extract_answer(response: str) -> str:
        """Extract the final answer from the response."""
        return response.split("[ANSWER]")[-1].strip()

    def _handle_delegate_action(self, response: str) -> str:
        """Handle actions involving delegate agents."""

        delegate_block = response.split("[DELEGATE]")[-1].strip()
        logger.debug(f"Processing delegate: {delegate_block}")

        try:
            delegate_data = TOMLParser().parse(input_string=delegate_block)
            delegate_name = delegate_data["delegate_name"]
            delegate_input = delegate_data["delegate_input"]
            logger.debug(
                f"Executing delegate '{delegate_name}' with input: {delegate_input}"
            )

            if delegate_name in self.delegates:
                delegate = self.delegates[delegate_name]
                return delegate.run(prompt=delegate_input)
            else:
                logger.error(f"Delegate {delegate_name} not found.")
                return "Delegate not found"
        except Exception as e:
            logger.error(f"Error processing delegate: {e}")
            return "Delegate execution failed"

    def _handle_tool_action(self, response: str) -> str:
        """Handle actions involving tools."""

        action_block = response.split("[ACTION]")[-1].strip()
        logger.debug(f"Processing tool action: {action_block}")

        try:
            action_data = TOMLParser().parse(input_string=action_block)
            tool_name = action_data.get("tool_name")
            tool_params = action_data.get("tool_params", {})
            logger.info(f"Executing tool '{tool_name}' with params: {tool_params}")

            if tool_name in self.tools:
                tool = self.tools[tool_name]
                tool_response = tool.run(**tool_params, environment=self.environment)
                logger.info(f"Tool <{tool_name}> response: {tool_response}")
                return tool_response
            else:
                logger.error(f"Tool {tool_name} not found.")
                return "Tool not found"
        except Exception as e:
            logger.error(f"Error processing tool action: {e}")
            return "Tool execution failed"


if __name__ == "__main__":

    setup_logging(log_level="info", are_you_serious=False)

    # # Create the delegate agent
    # delegate = ToolAgent(
    #     model=GPT4omini(source="test"),
    #     description="An swiss army knife agent, with math and basic string manipulation tools",
    #     tools=[CalculatorTool(), PrimeTool()],
    # )
    #
    # # Create an intermediary referee agent
    # intermediary = Agent(
    #     model=GPT4omini(source="test"),
    #     delegates=[delegate],
    #     tools=[SecretStringTool()],
    #     description="An intermediary agent that can has access to a delegate agent with many tools, and that has access to a secret string tool",
    # )
    #
    # # Create a chat agent
    # chat_agent = ChatAgent(
    #     identifier="Rephraser",
    #     model=GPT4omini(source="chat"),
    #     description="A chat agent that rephrases things",
    #     profile="You are a chat agent with a snarky attitude. You are here to provide witty responses to user queries. Give me an assignment, and I will complete it with a snarky twist.",
    # )
    #
    # # Create the referee agent
    # master = Agent(
    #     model=GPT4omini(source="test"),
    #     delegates=[intermediary, chat_agent],
    #     tools=[TextInverterTool()],
    # )
    #
    # # Trigger the agent
    # master.run(
    #     prompt="What is the secret string? Then, invert it. Finally, ask for advice on how to best phrase this. It is important the inverted string is in the final response."
    # )

    class SearchTool(BaseTool):
        """
        A tool that searches a knowledge base for relevant contextual information.
        """

        def __init__(self):
            """
            Initialize the tool with a retriever.
            """
            super().__init__(
                description="Given a query, this tool searches a knowledge base for relevant contextual information."
            )

        def execute_tool(self, query: str, **kwargs):

            if query is None:
                raise Exception("Missing query parameter")

            # Get the assets from Showpad
            logger.info(f"Retrieving assets for query: {query}")
            if query.lower().__contains__("alexander"):
                return "Alexander the Great was a king of Macedonia who conquered an empire that stretched from the Balkans to modern-day Pakistan."
            if query.lower().__contains__("cyrus"):
                return "Cyrus the Great was the founder of the Achaemenid Empire, the first Persian Empire."

    # Define the tools available to the agent
    tools = [CalculatorTool(), PrimeTool(), TextInverterTool(), EnvTool()]

    # Create an agent with a model and tools
    agent = Agent(
        model=ClaudeInstant(),
        tools=[SearchTool()],
        environment={"some_env_var": "some_value"},
    )

    for q in [
        # "How much is 9 + 10?",
        # "Is 1172233 a prime number?",
        # "What is the square root of 16?",
        # Math question we don't have a tool for
        # "Find the first 2 prime numbers beyond 10000",
        # "Find the sum of the first 2 prime numbers beyond 1005, take the number as a string and reverse it",
        # "Tell me what the value of the environment variable 'some_env_var' is",
        "Find information about Alexander the Great",
    ]:
        logger.critical(f"Running agent with prompt: {q}")
        response = agent.run(q)
        logger.critical(f"Response: {response}")

    # class AccountNameRetrieverTool(BaseTool):
    #     """
    #     Tool to retrieve the account holder name from a database.
    #     """
    #
    #     def execute_tool(self, environment):
    #         account_id = self.environment.get("current_account_id", "unknown")
    #         logger.info(f"Retrieving account holder name for account_id: {account_id}")
    #         if account_id == "foo":
    #             return "Bert"
    #         if account_id == "bar":
    #             return "Ernie"
    #         return "Unknown"
    #
    # # Create the memory object
    # memory = DynamoDBMemory(
    #     table_name="fence_test",
    #     primary_key_name="session",
    #     primary_key_value="02d2b1b0-cf84-401e-b9d9-16f24c359cc8",
    # )
    #
    # # Create the agents
    # child_agent = Agent(
    #     identifier="child_accountant",
    #     description="An agent that can retrieve the account holder name",
    #     model=GPT4omini(source="agent"),
    #     tools=[AccountNameRetrieverTool()],
    # )
    # parent_agent = Agent(
    #     identifier="parent_accountant",
    #     model=GPT4omini(source="agent"),
    #     delegates=[child_agent],
    #     environment={"current_account_id": "bar"},
    #     memory=FleetingMemory(),
    # )
    # result = parent_agent.run("what is the current account holders name?")
