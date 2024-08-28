from abc import ABC

import numexpr

from fence import (
    LLM,
    Link,
    Message,
    Messages,
    MessagesTemplate,
    TOMLParser,
    setup_logging,
)
from fence.models.gpt import GPT4o
from fence.prompts.agents import react_prompt

logger = setup_logging(__name__, log_level="info", serious_mode=False)


class Tool(ABC):

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
        for arg_name, arg_type in run_args.items():
            argument_string += (
                f"[[tools.tool_params]]\n"
                f'name = "{arg_name}"\n'
                f'type = "{arg_type.__name__}"\n'
            )

        toml_string = f"""[[tools]]
tool_name = "{self.__class__.__name__}"
tool_description = "{self.__doc__}"
{argument_string}"""

        return toml_string


class PrimeTool(Tool):
    """Checks if a number is prime"""

    def run(self, number: int) -> bool:
        """Check if the number is prime"""
        if number < 2:
            return False
        for i in range(2, number):
            if number % i == 0:
                return False
        return True


class CalculatorTool(Tool):
    """Perform mathematical calculations"""

    def run(self, expression: str) -> float | str:
        """Evaluate the mathematical expression"""
        try:

            result = numexpr.evaluate(expression)

        except Exception as e:
            logger.error(f"Error evaluating expression: {e}", exc_info=True)
            return f"Error evaluating expression {expression} - {e}"
        return result


class Memory(Messages):

    messages: list[Message] = []

    def add_message(self, role: str, content: str):

        if role == "system":
            self.system = content
        elif role in ["user", "assistant"]:
            self.messages.append(Message(role=role, content=content))
        else:
            raise ValueError(
                f"Role must be 'system', 'user', or 'assistant'. Got {role}"
            )


class Agent:
    """An LLM-based Agent, capable of using tools and models to generate responses"""

    def __init__(self, model: LLM | None = None, tools: list[Tool] | None = None):

        self.model = model
        self.tools = tools or []
        self.tool_names = [tool.__class__.__name__ for tool in self.tools]

        logger.info(
            f"Creating an agent with model: {model} and tools: {self.tool_names}"
        )

        self.context = Memory()
        self.context.add_message(role="system", content=react_prompt)

    def run(self, prompt: str) -> str:
        """Run the agent with the given prompt"""

        # Add the prompt to the history
        self.context.add_message(role="user", content=prompt)

        # Format tools for the prompt
        formatted_tools = "".join(tool.format_toml() for tool in self.tools)
        logger.debug(f"Tools: {formatted_tools}")

        response = None
        answer = None
        while response is None or not response.startswith("[ANSWER]"):

            # Base link
            link = Link(
                name="agent_step_link",
                model=self.model,
                template=MessagesTemplate(source=self.context),
            )

            response = link.run(input_dict={"tools": formatted_tools})["state"]
            logger.debug(f"Response: {response}")

            # Check if the response contains a tool action
            tool_response = None
            if "[ACTION]" in response:

                logger.debug("Found an action in the response")

                # Extract the action from the response
                action = response.split("[ACTION]")[-1].strip()
                logger.debug(f"Action: {action}")

                # Extract the tool name and parameters
                parser = TOMLParser()
                action_parsed = parser.parse(input_string=action)
                logger.debug(f"Action Parsed: {action_parsed}")

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
                    logger.info(
                        f"Checking if tool exists: {tool_name} in {self.tool_names} -> {tool_name in self.tool_names}"
                    )
                    if tool_name in self.tool_names:

                        logger.debug(
                            f"Running tool: {tool_name} with params: {tool_params}"
                        )
                        tool = self.tools[self.tool_names.index(tool_name)]
                        tool_response = tool.run(**tool_params)
                        logger.info(f"Tool Response: {tool_response}")

                        # Update the prompt with the tool response
                        prompt = tool_response
                    else:
                        prompt = "Invalid tool name"
                else:
                    prompt = "Invalid action"

            # Add the response to the history
            self.context.add_message(
                role="assistant",
                content=response
                + (f"[OBSERVATION] {tool_response}" if tool_response else ""),
            )

            # If the response contains an answer, break the loop
            if "[ANSWER]" in response:
                answer = response.split("[ANSWER]")[-1].strip()
                break

        return answer


if __name__ == "__main__":
    # Define the tools available to the agent
    tools = [CalculatorTool(), PrimeTool()]

    # Create an agent with a model and tools
    agent = Agent(model=GPT4o(source="agent"), tools=tools)

    for q in [
        "How much is 9 + 10?",
        "Is 1172233 a prime number?",
        "What is the square root of 16?",
    ]:
        logger.critical(f"Running agent with prompt: {q}")
        response = agent.run(q)
        logger.critical(f"Response: {response}")
