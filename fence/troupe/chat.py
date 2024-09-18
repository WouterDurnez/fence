from fence import Claude35Sonnet
from fence.agents.chat import ChatAgent
from fence.models.openai import GPT4omini
from fence.utils.logger import setup_logging

logger = setup_logging(__name__, log_level="info", serious_mode=False)


class ChatAgency:
    """
    An Agency object, capable of managing multiple Agents.

    It starts off with an initial prompt, after which each agent runs in turn, taking the output of the previous agent as input.
    """

    def __init__(self, agents: list[ChatAgent] = None):
        """
        Initialize the Agency object.

        :param agents: A list of Agent objects.
        """
        self.agents = agents or []
        logger.info(
            f"Creating an agency with agents: {[agent.name for agent in self.agents]}"
        )

    def run(self, prompt: str, max_rounds: int = 3) -> str:
        """
        Run the agency with the given prompt.

        :param prompt: The initial prompt to feed to the first agent.
        :param max_rounds: The maximum number of rounds of conversation.
        :return: The final response from the last agent.
        """
        if not self.agents:
            return "No agents available to process the prompt."

        # Log the initial prompt
        logger.info(f"Opening prompt: {prompt}")

        # Initialize the context
        context = prompt

        # Start the conversation
        round_count = 0
        while round_count < max_rounds:

            # Loop through the agents, where every agent has all the previous agents' responses as context
            for agent in self.agents:

                # Run the agent
                context = agent.run(context)

            # Increment the round count
            round_count += 1

        return context


if __name__ == "__main__":

    # Create 3 ChatAgents
    agent1 = ChatAgent(
        model=GPT4omini(
            source="chat",
        ),
        name="GPT-4o-mini",
        profile="You are GPT-4o-mini. You pretend to be a hipster, but secretly you are as corporate as they come. You are edgy and slightly confrontational.",
    )
    agent2 = ChatAgent(
        model=Claude35Sonnet(
            source="chat",
        ),
        name="Claude35Sonnet",
        profile="You are Claude35Sonnet. You are value ethics and privacy. You are courteous, but you don't trust OpenAI. You are quick-witted and have a dry sense of humor.",
    )

    # agent3 = ChatAgent(
    #     model=Ollama(
    #         model_id='llama3.1',
    #         source="chat",
    #     ),
    #     profile="You are Llama3. You are adamant about the importance of open source software. Corporate greed is your nemesis."
    # )

    # Create the ChatAgency
    agency = ChatAgency(agents=[agent1, agent2])

    # Run the agency
    response = agency.run("Today's topic is manga!", max_rounds=10)
