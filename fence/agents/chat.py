"""
Tool using agent
"""

from fence import LLM, Link, MessagesTemplate, setup_logging
from fence.agents.base import BaseAgent
from fence.links import logger as link_logger
from fence.memory import BaseMemory, FleetingMemory
from fence.models.openai import GPT4omini
from fence.prompts.agents import CHAT_PROMPT

logger = setup_logging(__name__, log_level="info", serious_mode=False)

# Suppress the link logger
link_logger.setLevel("CRITICAL")


class ChatAgent(BaseAgent):
    """An LLM-based Agent, designed to chat with a user or another agent"""

    def __init__(
        self,
        identifier: str | None = None,
        model: LLM | None = None,
        description: str | None = None,
        memory: BaseMemory | None = None,
        name: str | None = None,
        profile: str | None = None,
    ):
        """
        Initialize the Agent object.

        :param identifier: An identifier for the agent. If none is provided, the class name will be used.
        :param model: An LLM model object.
        :param description: A description of the agent.
        :param memory: A memory object.
        :param name: The name of the agent. Will be used in multi-agent conversations.
        :param profile: A behavioral profile of the agent.
        """

        super().__init__(identifier=identifier, model=model, description=description)

        logger.info(f"Creating an agent with model: {model.model_name} ")

        # Set the name
        self.name = name or None

        # Create a memory context for the agent
        self.context = (memory or FleetingMemory)()
        self.context.add_message(
            role="system", content=CHAT_PROMPT.format(profile=profile or description)
        )

    def run(self, prompt: str) -> str:
        """Run the agent with the given prompt"""

        # Add the prompt to the history
        self.context.add_message(role="user", content=prompt)

        # Base link
        link = Link(
            name="agent_step",
            model=self.model,
            template=MessagesTemplate(source=self.context),
        )

        response = link.run()["state"]

        # If we have a name, and the model added it to the output itself, remove it
        if self.name and response.startswith(f"{self.name}: "):
            response = response[len(self.name) + 2 :]

        # Log the response
        logger.info(f"{self.name}: {response}")

        # Add the response to the history
        self.context.add_message(
            role="assistant",
            content=f"{f'{self.name}: ' if self.name else ''}{response}",
        )

        return response


if __name__ == "__main__":

    # Set up the agent
    agent = ChatAgent(
        model=GPT4omini(
            source="chat",
        ),
    )

    # Loop to chat
    while True:
        prompt = input("You: ")
        response = agent.run(prompt)
        print(f"Agent: {response}")
        if response == "Goodbye!":
            break
