"""
Agent class to orchestrate a flow that potentially calls tools or other, specialized agents.
"""

import logging
import threading
from contextlib import contextmanager

from fence.agents.base import BaseAgent
from fence.links import Link
from fence.links import logger as link_logger
from fence.memory.base import BaseMemory, FleetingMemory
from fence.models.base import LLM
from fence.models.openai import GPT4omini
from fence.parsers import TOMLParser
from fence.prompts.agents import REACT_MULTI_AGENT_TOOL_PROMPT
from fence.tools.base import BaseTool
from fence.utils.logger import setup_logging

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
        prefill: str | None = None,
        log_agentic_response: bool = True,
        are_you_serious: bool = False,
        max_iterations: int = 5,
        iteration_timeout: float = 30.0,
        max_memory_size: int = 1000,
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
        :param prefill: A string (i.e., assistant message) to prefill the memory with.
        :param log_agentic_response: A flag to determine if the agent's responses should be logged.
        :param are_you_serious: A flag to determine if the log message should be printed in a frivolous manner.
        :param max_iterations: The maximum number of iterations to run before stopping the agent.
        :param iteration_timeout: The maximum time in seconds to run an iteration.
        :param max_memory_size: The maximum number of messages to store in memory.
        """
        super().__init__(
            identifier=identifier,
            model=model,
            description=description,
            memory=memory,
            environment=environment or {},
            prefill=prefill,
            log_agentic_response=log_agentic_response,
            are_you_serious=are_you_serious,
        )

        # Safeguards
        self.max_iterations = max_iterations
        self.iteration_timeout = iteration_timeout
        self.max_memory_size = max_memory_size

        # Set role
        self.role = (
            role
            or "You are a general purpose agent, capable of delegating tasks to other agents or tools."
        )

        # Delegates and tools initialization
        self.delegates = self._validate_entities(delegates or [])
        self.tools = self._validate_entities(tools or [])

        # Logging agent creation details
        logger.info(
            f"Initialized agent with model <{model.model_name}>, delegates: {list(self.delegates.keys())}, tools: {list(self.tools.keys())}"
        )

        # Prepare formatted delegates and tools
        self.formatted_delegates = self._format_entities(list(self.delegates.values()))
        self.formatted_tools = self._format_entities(list(self.tools.values()))

        # Dynamic system prompt generation
        self._update_system_message()

        # Initialize the memory buffer
        self._flush_memory()

    def _validate_entities(
        self, entities: list[BaseAgent | BaseTool]
    ) -> dict[str, BaseAgent | BaseTool]:
        """
        Validate and convert entities to a dictionary with robust error checking.

        :param entities: List of agents or tools
        :return: Dictionary of validated entities
        """
        validated_entities = {}
        for entity in entities:
            try:
                if not hasattr(entity, "run"):
                    logger.warning(
                        f"Entity {entity} does not have a 'run' method. Skipping."
                    )
                    continue

                # Add environment variables to the entity
                entity.environment.update(self.environment)

                # Use identifier or class name as key
                key = getattr(entity, "identifier", entity.__class__.__name__)
                validated_entities[key] = entity

            except Exception as e:
                logger.error(f"Error validating entity {entity}: {e}")

        return validated_entities

    def add_tool(self, tool: BaseTool) -> None:
        """
        Dynamically add a new tool to the agent.

        :param tool: Tool to add
        """
        if tool.__class__.__name__ not in self.tools:
            self.tools[tool.__class__.__name__] = tool
            self._update_system_message()

    def remove_tool(self, tool_name: str) -> None:
        """
        Remove a tool from the agent.

        :param tool_name: Name of the tool to remove
        """
        self.tools.pop(tool_name, None)
        self._update_system_message()

    def add_delegate(self, delegate: BaseAgent) -> None:
        """
        Dynamically add a new delegate to the agent.

        :param delegate: Delegate to add
        """
        if delegate.__class__.__name__ not in self.delegates:
            self.delegates[delegate.__class__.__name__] = delegate
            self._update_system_message()

    def remove_delegate(self, delegate_name: str) -> None:
        """
        Remove a delegate from the agent.

        :param delegate_name: Name of the delegate to remove
        """
        self.delegates.pop(delegate_name, None)
        self._update_system_message()

    def _update_system_message(self) -> None:
        """
        Dynamically update the system message based on current tools and delegates.
        """
        self._system_message = REACT_MULTI_AGENT_TOOL_PROMPT.format(
            role=self.role,
            delegates=self._format_entities(list(self.delegates.values())),
            tools=self._format_entities(list(self.tools.values())),
        )

        # Also update the system message in memory
        self.memory.set_system_message(self._system_message)

    def run(self, prompt: str, max_iterations: int | None = None) -> str:
        """
        Main execution loop to process the input prompt.

        :param prompt: The input prompt to process.
        :param max_iterations: Optional override for the maximum number of iterations.
        :return: The final answer from the agent.
        """

        # Truncate memory if it exceeds max size
        while len(self.memory.messages) > self.max_memory_size:
            self.memory.messages.pop(0)

        # Add initial prompt to the memory buffer
        self.memory.add_message(role="user", content=prompt)

        # Intialize variables pre-loop
        iteration_count, response, answer = 0, None, None

        # Let's go!
        while iteration_count < self.max_iterations:

            with self._timeout_handler():

                # Run the model with the current memory state
                link = Link(
                    name=f"{self.identifier or 'agent'}_step_{iteration_count}",
                    model=self.model,
                    template=self.memory.to_messages_template(),
                )
                response = link.run()["state"]
                logger.info(f"Model response: {response}")

                # Add the response to the memory buffer
                self.memory.add_message(role="assistant", content=response)

                # Extract the thought from the response for logging
                self._extract_thought(response)

                # Handle the response
                match response:

                    # Best case: we have an answer
                    case response if "[ANSWER]" in response:
                        answer = self._extract_answer(response)
                        break

                    # Handle delegate actions
                    case response if "[DELEGATE]" in response:
                        delegate_response = self._handle_delegate_action(response)
                        self.log(message=delegate_response, type="observation")
                        if delegate_response:
                            self.memory.add_message(
                                role="user",
                                content=f"[OBSERVATION] Call to delegate completed. Result: {delegate_response}",
                            )

                    # Handle tool actions
                    case response if "[ACTION]" in response:
                        tool_response = self._handle_tool_action(response)
                        self.log(message=tool_response, type="observation")
                        if tool_response:
                            self.memory.add_message(
                                role="user",
                                content=f"[OBSERVATION] Call to tool completed. Result: {tool_response}",
                            )

                iteration_count += 1

        self._flush_memory()
        return answer or "No answer found"

    def _extract_thought(self, response: str) -> str:
        """Extract the thought from the response. Starts with [THOUGHT], and ends before [ACTION], [DELEGATE], or [ANSWER]."""
        if "[THOUGHT]" not in response:
            logger.warning("No thought found in response")
            return ""

        thought = (
            response.split("[THOUGHT]")[1]
            .split("[ACTION]")[0]
            .split("[DELEGATE]")[0]
            .split("[ANSWER]")[0]
            .strip()
        )
        self.log(message=thought, type="thought")
        return thought

    def _extract_answer(self, response: str) -> str:
        """Extract the final answer from the response."""
        answer = response.split("[ANSWER]")[-1].strip()
        self.log(message=answer, type="answer")
        return answer

    def _handle_delegate_action(self, response: str) -> str:
        """Handle actions involving delegate agents."""

        delegate_block = response.split("[DELEGATE]")[-1].strip()
        self.log(message=delegate_block, type="delegation")
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
        self.log(message=action_block, type="action")
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

    ###################
    # Utility methods #
    ###################

    @staticmethod
    def _format_entities(entities: list) -> str:
        """Format delegates or tools into TOML representation."""
        return (
            "".join(entity.format_toml() for entity in entities)
            if entities
            else "None available"
        )

    @contextmanager
    def _timeout_handler(self):
        def interrupt():
            raise TimeoutError("Timeout!")

        timer = threading.Timer(self.iteration_timeout, interrupt)
        try:
            timer.start()
            yield
        finally:
            timer.cancel()


TimeoutError("Iteration exceeded maximum time")


if __name__ == "__main__":

    setup_logging(log_level="debug", are_you_serious=False)

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

    # class SearchTool(BaseTool):
    #     """
    #     A tool that searches a knowledge base for relevant contextual information.
    #     """

    #     def __init__(self):
    #         """
    #         Initialize the tool with a retriever.
    #         """
    #         super().__init__(
    #             description="Given a query, this tool searches a knowledge base for relevant contextual information."
    #         )

    #     def execute_tool(self, query: str, **kwargs):

    #         if query is None:
    #             raise Exception("Missing query parameter")

    #         # Get the assets from Showpad
    #         logger.info(f"Retrieving assets for query: {query}")
    #         if query.lower().__contains__("alexander"):
    #             return "Alexander the Great was a king of Macedonia who conquered an empire that stretched from the Balkans to modern-day Pakistan."
    #         if query.lower().__contains__("cyrus"):
    #             return "Cyrus the Great was the founder of the Achaemenid Empire, the first Persian Empire."

    # # Define the tools available to the agent
    # tools = [CalculatorTool(), PrimeTool(), TextInverterTool(), EnvTool()]

    # # Create an agent with a model and tools
    # agent = Agent(
    #     model=ClaudeInstant(),
    #     tools=[SearchTool()],
    #     environment={"some_env_var": "some_value"},
    # )

    # for q in [
    #     # "How much is 9 + 10?",
    #     # "Is 1172233 a prime number?",
    #     # "What is the square root of 16?",
    #     # Math question we don't have a tool for
    #     # "Find the first 2 prime numbers beyond 10000",
    #     # "Find the sum of the first 2 prime numbers beyond 1005, take the number as a string and reverse it",
    #     # "Tell me what the value of the environment variable 'some_env_var' is",
    #     "Find information about Alexander the Great",
    # ]:
    #     logger.critical(f"Running agent with prompt: {q}")
    #     response = agent.run(q)
    #     logger.critical(f"Response: {response}")

    class AccountNameRetrieverTool(BaseTool):
        """
        Tool to retrieve the account holder name from a database.
        """

        def execute_tool(self, environment):
            account_id = self.environment.get("current_account_id", "unknown")
            logger.info(f"Retrieving account holder name for account_id: {account_id}")
            if account_id == "foo":
                return "Bert"
            if account_id == "bar":
                return "Ernie"
            return "Unknown"

    # Create the memory object
    # memory = DynamoDBMemory(
    #     table_name="fence_test",
    #     primary_key_name="session",
    #     primary_key_value="02d2b1b0-cf84-401e-b9d9-16f24c359cc8",
    # )

    # Create the agents
    child_agent = Agent(
        identifier="child_accountant",
        description="An agent that can retrieve the account holder name",
        model=GPT4omini(source="agent"),
        tools=[AccountNameRetrieverTool()],
    )
    parent_agent = Agent(
        identifier="parent_accountant",
        model=GPT4omini(source="agent"),
        delegates=[child_agent],
        environment={"current_account_id": "bar"},
        memory=FleetingMemory(),
    )
    result = parent_agent.run("what is the current account holders name?")
