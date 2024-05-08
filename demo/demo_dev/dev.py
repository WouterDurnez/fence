# Relevancy template
from fence import MessagesTemplate, Link, ClaudeHaiku
from fence.models.claude3 import Claude3Base
from fence.parsers import BoolParser
from fence.templates.models import Message, Messages
from fence.utils.base import setup_logging

logger = setup_logging(__name__)

model = ClaudeHaiku(source="dev", region="us-east-1")


def get_first_n_words(text: str, n: int) -> str:
    """
    Get the first n words from a text
    """
    return " ".join(text.split()[:n])


def get_word_count(text: str) -> int:
    """
    Get the word count of a text
    """
    return len(text.split())


class LLMHelper:
    """
    Helper class to perform basic LLM tasks
    """

    def __init__(self, model: Claude3Base = None):
        self.model = (
            model if model else ClaudeHaiku(source="llm_helper", region="us-east-1")
        )

    def check_relevancy(self, text: str, topic: str):
        """
        Check the relevancy of a text to a given topic
        :param text: The text to check
        :param topic: The topic to check against
        """

        # Define template
        relevancy_template = MessagesTemplate(
            source=Messages(
                system="""You are a helpful search assistant. Your job is to judge 
                whether a given text is relevant to a given topic. You are given a text 
                and a topic. You should return True if the text is relevant to the 
                topic, and False otherwise.""",
                messages=[
                    Message(
                        role="user",
                        content="""You are given the following text: {{text}}
                    The topic is: {{topic}}
                    Is the text relevant to the topic? Only reply with True or False.
                    """,
                    ),
                    Message(
                        role="assistant",
                        content="Evaluation:",
                    ),
                ],
            )
        )
        logger.info(
            f"Checking relevancy of text <{get_first_n_words(text, 10)}...> to topic <{topic}>"
        )
        relevancy_link = Link(
            template=relevancy_template,
            llm=self.model,
            name="relevancy_checker",
            output_key="is_relevant",
            parser=BoolParser(),
        )
        response = relevancy_link.run({"text": text, "topic": topic})

        # Return check
        relevancy_check = response["is_relevant"]
        logger.info(f"Relevancy check response: {response} -> {relevancy_check}")

        return relevancy_check


checker = LLMHelper()
text = "The spoon was invented in 1000 BC."
topic = "history"
relevancy_check = checker.check_relevancy
is_relevant = relevancy_check(text, topic)
print(is_relevant)
