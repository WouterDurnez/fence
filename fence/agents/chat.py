"""
Tool using agent
"""

import logging

from fence.agents.base import BaseAgent
from fence.links import Link
from fence.links import logger as link_logger
from fence.memory.base import BaseMemory, FleetingMemory
from fence.models.base import LLM
from fence.models.openai import GPT4omini
from fence.prompts.agents import CHAT_PROMPT

logger = logging.getLogger(__name__)


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
        role: str | None = None,
    ):
        """
        Initialize the Agent object.

        :param identifier: An identifier for the agent. If none is provided, the class name will be used.
        :param model: An LLM model object.
        :param description: A description of the agent.
        :param memory: A memory object.
        :param name: The name of the agent. Will be used in multi-agent conversations.
        :param role: A behavioral profile of the agent.
        """

        super().__init__(identifier=identifier, model=model, description=description)

        logger.info(f"Creating an agent with model: {model.model_name} ")

        # Set the name
        self.name = name or None

        # Create a memory context for the agent
        self.memory = memory or FleetingMemory()

        # Set system message
        self._system_message = CHAT_PROMPT.format(role=role or description)

        # Flush memory
        self._flush_memory()

    def run(self, prompt: str) -> str:
        """Run the agent with the given prompt"""

        # Add the prompt to the history
        self.memory.add_message(role="user", content=prompt)

        # Base link
        link = Link(
            name="agent_step",
            model=self.model,
            template=self.memory.to_messages_template(),
        )

        response = link.run()["state"]

        # If we have a name, and the model added it to the output itself, remove it
        if self.name and response.startswith(f"{self.name}: "):
            response = response[len(self.name) + 2 :]

        # Log the response
        logger.info(f"{self.name}: {response}")

        # Add the response to the history
        self.memory.add_message(
            role="assistant",
            content=f"{f'{self.name}: ' if self.name else ''}{response}",
        )

        return response


if __name__ == "__main__":

    content = """Nikola Tesla (/ˈtɛslə/;[2] Serbian Cyrillic: Никола Тесла, [nǐkola têsla]; 10 July 1856[a] – 7 January 1943) was a Serbian-American[3][4] engineer, futurist, and inventor. He is known for his contributions to the design of the modern alternating current (AC) electricity supply system.[5]

Born and raised in the Austrian Empire, Tesla first studied engineering and physics in the 1870s without receiving a degree. He then gained practical experience in the early 1880s working in telephony and at Continental Edison in the new electric power industry. In 1884 he immigrated to the United States, where he became a naturalized citizen. He worked for a short time at the Edison Machine Works in New York City before he struck out on his own. With the help of partners to finance and market his ideas, Tesla set up laboratories and companies in New York to develop a range of electrical and mechanical devices. His AC induction motor and related polyphase AC patents, licensed by Westinghouse Electric in 1888, earned him a considerable amount of money and became the cornerstone of the polyphase system which that company eventually marketed.

Attempting to develop inventions he could patent and market, Tesla conducted a range of experiments with mechanical oscillators/generators, electrical discharge tubes, and early X-ray imaging. He also built a wirelessly controlled boat, one of the first ever exhibited. Tesla became well known as an inventor and demonstrated his achievements to celebrities and wealthy patrons at his lab, and was noted for his showmanship at public lectures. Throughout the 1890s, Tesla pursued his ideas for wireless lighting and worldwide wireless electric power distribution in his high-voltage, high-frequency power experiments in New York and Colorado Springs. In 1893, he made pronouncements on the possibility of wireless communication with his devices. Tesla tried to put these ideas to practical use in his unfinished Wardenclyffe Tower project, an intercontinental wireless communication and power transmitter, but ran out of funding before he could complete it.

After Wardenclyffe, Tesla experimented with a series of inventions in the 1910s and 1920s with varying degrees of success. Having spent most of his money, Tesla lived in a series of New York hotels, leaving behind unpaid bills. He died in New York City in January 1943.[6] Tesla's work fell into relative obscurity following his death, until 1960, when the General Conference on Weights and Measures named the International System of Units (SI) measurement of magnetic flux density the tesla in his honor. There has been a resurgence in popular interest in Tesla since the 1990s.[7]"""

    # Set up the agent
    agent = ChatAgent(
        model=GPT4omini(
            source="chat",
        ),
        role=f"""You are a helpful assistant, capable of answering questions about a document. You do not deviate from this task. You should therefore not talk about anything other than this content. You will be given the content, i.e. the transcript of what is in it.

                The content you have access to is this:

                <content>
                {content}
                </content>

                Again, you can only answer questions about the content you have access to.
                """,
    )

    # Loop to chat
    while True:
        prompt = input("You: ")
        response = agent.run(prompt)
        print(f"Agent: {response}")
        if response == "Goodbye!":
            break
