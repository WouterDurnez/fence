"""
Agent class to orchestrate a flow that potentially calls tools or other, specialized agents.
"""

from fence import LLM, Link, MessagesTemplate, TOMLParser, setup_logging
from fence.agents.base import BaseAgent
from fence.agents.tool import ToolAgent
from fence.links import logger as link_logger
from fence.memory import BaseMemory, FleetingMemory
from fence.models.openai import GPT4omini
from fence.prompts.agents import REACT_MULTI_AGENT_PROMPT
from fence.tools.math import CalculatorTool, PrimeTool
from fence.tools.text import TextInverterTool

logger = setup_logging(__name__, log_level="info", serious_mode=False)

# Suppress the link logger
link_logger.setLevel("INFO")


class RefereeAgent(BaseAgent):
    """An LLM-based Agent, capable of delegating to other agents"""

    def __init__(
        self,
        model: LLM = None,
        description: str | None = None,
        delegates: list[BaseAgent] = None,
        memory: BaseMemory = None,
    ):
        """
        Initialize the Referee agent object.

        :param model: An LLM model object.
        :param delegates: A list of Agent objects, to which the Referee can delegate.
        :param memory: A memory class.
        """

        super().__init__(model=model, description=description)

        # Store the tools and their names
        self.delegates = delegates or []
        self.delegate_names = [agent.__class__.__name__ for agent in self.delegates]

        logger.info(
            f"Creating an agent with model <{model.model_name}> and delegates: {self.delegate_names}"
        )

        # Create a memory context for the agent
        self.memory = memory
        self.context = None
        self.wipe_context(memory=self.memory)

        # Format tools for the prompt
        self.formatted_agents = "".join(agent.format_toml() for agent in self.delegates)
        logger.debug(f"Tools: {self.formatted_agents}")

    def run(self, prompt: str) -> str:
        """Run the agent with the given prompt"""

        # Add the prompt to the history
        self.context.add_message(role="user", content=prompt)

        ################
        # Agentic loop #
        ################
        response = None
        answer = None
        iteration_count = 0
        max_iterations = 30  # safeguard against infinite loops

        while response is None or not response.startswith("[ANSWER]"):

            # Increment iteration counter and check for loop limit
            iteration_count += 1
            if iteration_count > max_iterations:
                logger.warning("Max iterations reached, exiting loop.")
                break

            # Base link
            link = Link(
                name="referee_step",
                model=self.model,
                template=MessagesTemplate(source=self.context),
            )

            response = link.run(input_dict={"agents": self.formatted_agents})["state"]
            logger.info(f"Model response: {response}")

            # Add the response to the history
            self.context.add_message(role="assistant", content=response)

            # If the response contains an answer, break the loop
            if "[ANSWER]" in response:
                answer = response.split("[ANSWER]")[-1].strip()
                break

            # Check if the response contains a tool action
            delegate_response = None
            if "[DELEGATE]" in response:

                logger.debug("Found an agent in the response")

                # Extract the action from the response
                delegate = response.split("[DELEGATE]")[-1].strip()
                logger.debug(f"Delegate: {delegate}")

                # Extract the tool name and parameters
                parser = TOMLParser()
                try:
                    delegate_parsed = parser.parse(input_string=delegate)
                    logger.debug(f"Delegate Parsed: {delegate_parsed}")
                except Exception as e:
                    logger.error(f"Failed to parse delegate agent: {e}")
                    prompt = "Invalid delegate"
                    self.context.add_message(
                        role="assistant", content="Invalid delegate"
                    )
                    continue

                # If the action is valid, run the tool
                if (
                    delegate_parsed
                    and "agent_name" in delegate_parsed
                    and "agent_input" in delegate_parsed
                ):
                    delegate_name = delegate_parsed["agent_name"]
                    delegate_input = delegate_parsed["agent_input"]
                    logger.info(
                        f"Executing tool: {delegate_name} with input: {delegate_input}"
                    )

                    # Check if the tool exists
                    if delegate_name in self.delegate_names:
                        logger.debug(
                            f"Running tool: {delegate_name} with input: {delegate_input}"
                        )
                        delegate = self.delegates[
                            self.delegate_names.index(delegate_name)
                        ]
                        try:
                            delegate_response = delegate.run(prompt=delegate_input)
                            logger.info(f"Delegate Response: {delegate_response}")
                        except Exception as e:
                            logger.error(f"Error running delegate {delegate_name}: {e}")
                            delegate_response = "Delegate execution failed"
                    else:
                        logger.error(f"Delegate {delegate_name} not found")
                        delegate_response = "Delegate not found"
                        continue
                else:
                    prompt = "Invalid delegate"
                    self.context.add_message(
                        role="assistant", content="Invalid delegate"
                    )
                    continue

            # Add the response to the history
            self.context.add_message(
                role="user",
                content=(
                    f"[OBSERVATION] {delegate_response}"
                    if delegate_response is not None
                    else "No delegate response"
                ),
            )

        # Clear the memory context
        self.wipe_context(self.memory)

        return answer

    def wipe_context(self, memory: BaseMemory | None = None):
        """Clear the memory context"""
        self.context = (
            memory if isinstance(memory, BaseMemory) else (memory or FleetingMemory)()
        )
        self.context.add_message(role="system", content=REACT_MULTI_AGENT_PROMPT)


if __name__ == "__main__":

    # Create the delegate agent
    delegate = ToolAgent(
        model=GPT4omini(source="test"),
        description="An swiss army knife agent, with math and basic string manipulation tools",
        tools=[CalculatorTool(), PrimeTool(), TextInverterTool()],
    )

    # Create an intermediary referee agent
    intermediary = RefereeAgent(
        model=GPT4omini(source="test"),
        delegates=[delegate],
        description="An intermediary agent that can delegate to a tool agent",
    )

    # Create the referee agent
    master = RefereeAgent(model=GPT4omini(source="test"), delegates=[intermediary])

    # Trigger the agent
    master.run(prompt="What is 2+2?")
