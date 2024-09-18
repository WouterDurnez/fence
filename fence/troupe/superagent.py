"""
Agent class to orchestrate a flow that potentially calls tools or other, specialized agents.
"""

from fence import LLM, Link, MessagesTemplate, TOMLParser, setup_logging
from fence.agents.base import BaseAgent
from fence.agents.chat import ChatAgent
from fence.agents.tool import ToolAgent
from fence.links import logger as link_logger
from fence.memory import BaseMemory, FleetingMemory
from fence.models.openai import GPT4omini
from fence.prompts.agents import REACT_MULTI_AGENT_TOOL_PROMPT
from fence.tools.base import BaseTool
from fence.tools.math import CalculatorTool, PrimeTool
from fence.tools.temp import SecretStringTool
from fence.tools.text import TextInverterTool

logger = setup_logging(__name__, log_level="info", serious_mode=False)

# Suppress the link logger
link_logger.setLevel("INFO")


class SuperAgent(BaseAgent):
    """An LLM-based agent capable of delegating tasks to other agents or tools."""

    def __init__(
        self,
        identifier: str | None = None,
        model: LLM | None = None,
        description: str | None = None,
        delegates: list[BaseAgent] | None = None,
        tools: list[BaseTool] | None = None,
        memory: BaseMemory | None = None,
    ):
        """
        Initialize the Agent object.

        :param identifier: An identifier for the agent. If none is provided, the class name will be used.
        :param model: An LLM model object.
        :param description: A description of the agent.
        :param delegates: A list of delegate agents.
        :param tools: A list of Tool objects.
        :param memory: A memory class.
        """
        super().__init__(identifier=identifier, model=model, description=description)

        # Delegates and tools initialization
        self.delegates = (
            {delegate.identifier: delegate for delegate in delegates}
            if delegates
            else {}
        )
        self.tools = {tool.__class__.__name__: tool for tool in tools} if tools else {}

        # Logging agent creation details
        logger.info(
            f"Initialized agent with model <{model.model_name}>, delegates: {self.delegates.keys()}, tools: {self.tools.keys()}"
        )

        # Memory setup
        self.memory = memory or FleetingMemory()
        self.context = None
        self.wipe_context()

        # Prepare formatted delegates and tools
        self.formatted_delegates = self._format_entities(list(self.delegates.values()))
        self.formatted_tools = self._format_entities(list(self.tools.values()))

    def run(self, prompt: str) -> str:
        """
        Main execution loop to process the input prompt.

        :param prompt: The input prompt to process.
        :return: The final answer from the agent.
        """

        # Add initial prompt to the context
        self.context.add_message(role="user", content=prompt)

        max_iterations, iteration_count = 30, 0
        response, answer = None, None

        while iteration_count < max_iterations:

            link = Link(
                name=f"referee_step_{iteration_count}",
                model=self.model,
                template=MessagesTemplate(source=self.context),
            )
            response = link.run(
                input_dict={
                    "delegates": self.formatted_delegates,
                    "tools": self.formatted_tools,
                }
            )["state"]
            logger.info(f"Model response: {response}")

            self.context.add_message(role="assistant", content=response)

            if "[ANSWER]" in response:
                answer = self._extract_answer(response)
                break

            if "[DELEGATE]" in response:
                delegate_response = self._handle_delegate_action(response)
                if delegate_response:
                    self.context.add_message(
                        role="user", content=f"[OBSERVATION] {delegate_response}"
                    )

            if "[ACTION]" in response:
                tool_response = self._handle_tool_action(response)
                if tool_response:
                    self.context.add_message(
                        role="user", content=f"[OBSERVATION] {tool_response}"
                    )

            iteration_count += 1

        self.wipe_context()
        return answer or "No answer found"

    def wipe_context(self):
        """Clear or reset the agent's memory context."""
        self.context = self.memory or FleetingMemory()
        self.context.add_message(role="system", content=REACT_MULTI_AGENT_TOOL_PROMPT)

    def _format_entities(self, entities: list) -> str:
        """Format delegates or tools into TOML representation."""
        return (
            "".join(entity.format_toml() for entity in entities)
            if entities
            else "None available"
        )

    def _extract_answer(self, response: str) -> str:
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
            logger.info(
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
                return tool.run(**tool_params)
            else:
                logger.error(f"Tool {tool_name} not found.")
                return "Tool not found"
        except Exception as e:
            logger.error(f"Error processing tool action: {e}")
            return "Tool execution failed"


if __name__ == "__main__":

    # Create the delegate agent
    delegate = ToolAgent(
        model=GPT4omini(source="test"),
        description="An swiss army knife agent, with math and basic string manipulation tools",
        tools=[CalculatorTool(), PrimeTool()],
    )

    # Create an intermediary referee agent
    intermediary = SuperAgent(
        model=GPT4omini(source="test"),
        delegates=[delegate],
        tools=[SecretStringTool()],
        description="An intermediary agent that can has access to a delegate agent with many tools, and that has access to a secret string tool",
    )

    # Create a chat agent
    chat_agent = ChatAgent(
        identifier="Rephraser",
        model=GPT4omini(source="chat"),
        description="A chat agent that rephrases things",
        profile="You are a chat agent with a snarky attitude. You are here to provide witty responses to user queries. Give me an assignment, and I will complete it with a snarky twist.",
    )

    # Create the referee agent
    master = SuperAgent(
        model=GPT4omini(source="test"),
        delegates=[intermediary, chat_agent],
        tools=[TextInverterTool()],
    )

    # Trigger the agent
    master.run(
        prompt="What is the secret string? Then, invert it. Finally, ask for advice on how to best phrase this. It is important the inverted string is in the final response."
    )
