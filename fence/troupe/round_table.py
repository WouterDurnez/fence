"""
RoundTable class for managing conversations between multiple ChatAgents
"""

import logging
from typing import List

from pydantic import BaseModel

from fence import ClaudeHaiku, setup_logging
from fence.agents import ChatAgent
from fence.agents.chat import logger as chat_agent_logger
from fence.links import logger as link_logger
from fence.memory.base import BaseMemory, FleetingMemory
from fence.models.openai import GPT4omini

logger = logging.getLogger(__name__)

# Suppress loggers
link_logger.setLevel("CRITICAL")
chat_agent_logger.setLevel("CRITICAL")


class ConversationTurn(BaseModel):
    """Represents a single turn in the conversation"""

    agent_name: str
    message: str

    def __str__(self):
        return f"{self.agent_name}: {self.message}"


class RoundTable:
    """
    A RoundTable object that manages multiple Agents in a conversation.

    It maintains a complete conversation history and ensures each agent has
    access to the full context when responding.
    """

    def __init__(
        self,
        agents: list[ChatAgent] = None,
        memory: BaseMemory | None = None,
    ):
        """
        Initialize the RoundTable.

        :param agents: A list of ChatAgent objects
        :param memory: Optional memory store for the conversation
        """
        self.agents = agents or []
        self.memory = memory or FleetingMemory()
        self.conversation_history: List[ConversationTurn] = []

        logger.info(
            f"Creating a round table with agents: {[agent.name for agent in self.agents]}"
        )

    def _format_conversation_history(self) -> str:
        """Format the entire conversation history into a readable string"""
        formatted_history = []

        # Add each turn to the formatted history
        for turn in self.conversation_history:
            formatted_history.append(str(turn))

        return "\n".join(formatted_history)

    def _create_agent_context(self, current_history: str) -> str:
        """Create the context message that will be sent to the next agent"""
        return (
            "This is the conversation so far:\n\n"
            f"{current_history}\n\n"
            "Please provide your response, continuing the conversation naturally. "
            "Remember to stay in character and address the ongoing discussion."
        )

    def add_turn(self, agent_name: str, message: str):
        """Add a new turn to the conversation history"""
        turn = ConversationTurn(agent_name=agent_name, message=message)
        self.conversation_history.append(turn)

        # Also store in persistent memory if needed
        self.memory.add_message(
            role="assistant" if agent_name != "System" else "system", content=str(turn)
        )

    def run(self, prompt: str, max_rounds: int = 3) -> str:
        """
        Run the round table discussion with the given prompt.

        :param prompt: The initial prompt to start the conversation
        :param max_rounds: Maximum number of rounds of conversation
        :return: The complete conversation history as a string
        """
        if not self.agents:
            return "No agents available to process the prompt."

        # Log the initial prompt
        logger.info(f"Opening prompt: {prompt}")

        # Add the initial prompt to the history
        self.add_turn("System", prompt)

        # Run the conversation for the specified number of rounds
        for round_num in range(max_rounds):
            logger.debug(f"Starting round {round_num + 1}")

            # Each agent takes a turn
            for agent in self.agents:

                # Get the full conversation history
                current_history = self._format_conversation_history()

                # Create the context for this agent
                context = self._create_agent_context(current_history)

                # Get the agent's response
                response = agent.run(context)

                # Add the response to the history
                self.add_turn(agent.name, response)

                logger.info(f"{agent.name}: {response}")

        # Return the complete conversation history
        return self._format_conversation_history()

    def save_transcript(self, filename: str):
        """Save the conversation transcript to a file"""
        with open(filename, "w") as f:
            f.write(self._format_conversation_history())


if __name__ == "__main__":

    setup_logging(log_level="info", are_you_serious=False)

    # Create some agents
    agent1 = ChatAgent(
        name="Detective 🕵️",
        role="You are a detective trying to solve a mystery. You ask probing questions and analyze details carefully.",
        model=GPT4omini(),
    )

    agent2 = ChatAgent(
        name="Witness 👀",
        role="You are a witness to a crime. You're nervous and sometimes contradictory, but you want to help.",
        model=ClaudeHaiku(),
    )

    agent3 = ChatAgent(
        name="Psychologist 🧑‍⚕️",
        role="You are a forensic psychologist. You analyze the behavioral aspects of the conversation and provide insights.",
        model=ClaudeHaiku(),
    )

    # Create the round table
    round_table = RoundTable(agents=[agent1, agent2, agent3])

    # Start the conversation
    transcript = round_table.run(
        prompt="A valuable painting has been stolen from the museum. Let's investigate what happened.",
        max_rounds=2,
    )
