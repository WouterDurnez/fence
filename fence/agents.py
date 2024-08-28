from fence import LLM, Link, MessagesTemplate, TOMLParser, setup_logging
from fence.links import logger as link_logger
from fence.memory import FleetingMemory
from fence.models.gpt import GPT4o
from fence.prompts.agents import react_prompt2
from fence.tools.base import BaseTool
from fence.tools.math import CalculatorTool, PrimeTool

logger = setup_logging(__name__, log_level="info", serious_mode=False)

# Suppress the link logger
link_logger.setLevel("INFO")


class Agent:
    """An LLM-based Agent, capable of using tools and models to generate responses"""

    def __init__(self, model: LLM = None, tools: list[BaseTool] = None):
        """
        Initialize the Agent object.

        :param model: An LLM model object.
        :param tools: A list of Tool objects.
        """
        self.model = model
        self.tools = tools or []
        self.tool_names = [tool.__class__.__name__ for tool in self.tools]

        logger.info(
            f"Creating an agent with model: {model} and tools: {self.tool_names}"
        )

        # Create a memory context for the agent
        self.context = FleetingMemory()
        self.context.add_message(role="system", content=react_prompt2)

        # Format tools for the prompt
        self.formatted_tools = "".join(tool.format_toml() for tool in self.tools)
        logger.debug(f"Tools: {self.formatted_tools}")

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
                name="agent_step",
                model=self.model,
                template=MessagesTemplate(source=self.context),
            )

            response = link.run(input_dict={"tools": self.formatted_tools})["state"]
            logger.info(f"Model response: {response}")

            # Add the response to the history
            self.context.add_message(role="assistant", content=response)

            # If the response contains an answer, break the loop
            if "[ANSWER]" in response:
                answer = response.split("[ANSWER]")[-1].strip()
                break

            # Check if the response contains a tool action
            tool_response = None
            if "[ACTION]" in response:

                logger.debug("Found an action in the response")

                # Extract the action from the response
                action = response.split("[ACTION]")[-1].strip()
                logger.debug(f"Action: {action}")

                # Extract the tool name and parameters
                parser = TOMLParser()
                try:
                    action_parsed = parser.parse(input_string=action)
                    logger.debug(f"Action Parsed: {action_parsed}")
                except Exception as e:
                    logger.error(f"Failed to parse action: {e}")
                    prompt = "Invalid action"
                    self.context.add_message(role="assistant", content="Invalid action")
                    continue

                # If the action is valid, run the tool
                if (
                    action_parsed
                    and "tool_name" in action_parsed
                    and "tool_params" in action_parsed
                ):
                    tool_name = action_parsed["tool_name"]
                    tool_params = action_parsed["tool_params"]
                    logger.info(
                        f"Executing tool: {tool_name} with params: {tool_params}"
                    )

                    # Check if the tool exists
                    if tool_name in self.tool_names:
                        logger.debug(
                            f"Running tool: {tool_name} with params: {tool_params}"
                        )
                        tool = self.tools[self.tool_names.index(tool_name)]
                        try:
                            tool_response = tool.run(**tool_params)
                            logger.info(f"Tool Response: {tool_response}")
                        except Exception as e:
                            logger.error(f"Error running tool {tool_name}: {e}")
                            tool_response = "Tool execution failed"
                    else:
                        logger.error(f"Tool {tool_name} not found")
                        prompt = "Invalid tool name"
                        self.context.add_message(
                            role="assistant", content="Invalid tool name"
                        )
                        continue
                else:
                    prompt = "Invalid action"
                    self.context.add_message(role="assistant", content="Invalid action")
                    continue

            # Add the response to the history
            self.context.add_message(
                role="user",
                content=(
                    f"[OBSERVATION] {tool_response}"
                    if tool_response is not None
                    else "No tool response"
                ),
            )

        # Clear the memory context
        self.clear_memory()

        return answer

    def clear_memory(self):
        """Clear the memory context"""
        self.context = FleetingMemory()
        self.context.add_message(role="system", content=react_prompt2)


if __name__ == "__main__":
    # Define the tools available to the agent
    tools = [CalculatorTool(), PrimeTool()]

    # Create an agent with a model and tools
    agent = Agent(model=GPT4o(source="agent"), tools=tools)

    for q in [
        # "How much is 9 + 10?",
        # "Is 1172233 a prime number?",
        # "What is the square root of 16?",
        # Math question we don't have a tool for
        "Find the first 2 prime numbers beyond 10000",
        "Find the sum of the first 2 prime numbers beyond 10000",
    ]:
        logger.critical(f"Running agent with prompt: {q}")
        response = agent.run(q)
        logger.critical(f"Response: {response}")
