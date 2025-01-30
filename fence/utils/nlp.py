"""
Helper tools to perform basic LLM-related tasks

- Relevancy check: Check the relevancy of a text to a given topic
- TODO: Consistency/contradiction check: Check the contradiction of two texts
"""

import logging
from math import ceil

from fence.links import Link
from fence.models.base import LLM
from fence.models.bedrock.claude import ClaudeHaiku
from fence.parsers import BoolParser, TripleBacktickParser
from fence.templates.messages import MessagesTemplate
from fence.templates.models import Message, Messages
from fence.utils.optim import retry

logger = logging.getLogger(__name__)

DEFAULT_REFERENCE_FLAGS = ["according", "text", "passage"]

##########
# String #
##########


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


############
# Chunking #
############


class TextChunker:
    def __init__(
        self,
        chunk_size: int,
        overlap: float | None = 0.01,
        text: str = None,
    ):
        self.text = text
        self.chunk_size = chunk_size
        self.overlap = overlap

    @staticmethod
    def _get_approximate_chunk_size(text: str, chunk_size: int) -> int:
        """
        Get approximate chunk size for a given text and chunk size
        :param text: text string
        :param chunk_size: desired chunk size target
        :return: recalculated chunk size
        """
        approx_chunk_count = ceil(len(text) / chunk_size)
        approx_chunk_size = ceil(len(text) / approx_chunk_count)
        return approx_chunk_size

    def split_text(self, text: str = None) -> list[str]:
        text = text if text is not None else self.text

        core_chunk_size = self._get_approximate_chunk_size(
            text=text, chunk_size=int(self.chunk_size / (1 + self.overlap))
        )

        chunk_size_with_overlap = (
            (core_chunk_size + int(core_chunk_size * self.overlap))
            if self.overlap
            else core_chunk_size
        )

        logger.debug(
            f"Chunking {len(text)} characters into {self.chunk_size=}-sized chunks with : {core_chunk_size=}, {chunk_size_with_overlap=}"
        )

        remaining_text = text
        chunks = []

        while len(remaining_text) > core_chunk_size:
            chunk = (
                remaining_text[:chunk_size_with_overlap]
                if len(remaining_text) > chunk_size_with_overlap
                else remaining_text
            )
            chunks.append(chunk)
            remaining_text = remaining_text[core_chunk_size:]

        if remaining_text:
            chunks.append(remaining_text)

        logger.debug(
            f"Chunking complete. {len(chunks)} chunks created. Smallest chunk size: {min([len(chunk) for chunk in chunks])}"
        )

        return chunks


#######
# LLM #
#######


class LLMHelper:
    """
    Helper class to perform basic LLM tasks
    """

    def __init__(self, model: LLM = None):
        self.model = model if model else ClaudeHaiku(source="llm_helper")

    @retry(max_retries=3)
    def check_relevancy(self, text: str, topic: str) -> bool:
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
                        content="""You are given the following text: {text}
                    The topic is: {topic}
                    Is the text relevant to the topic? Only reply with True or False, and nothing else.
                    """,
                    ),
                    Message(
                        role="assistant",
                        content="Evaluation boolean (True/False):",
                    ),
                ],
            )
        )
        logger.debug(
            f"Checking relevancy of text <{get_first_n_words(text=text, n=10)}...> to topic <{topic}>"
        )
        relevancy_link = Link(
            template=relevancy_template,
            model=self.model,
            name="relevancy_checker",
            output_key="is_relevant",
            parser=BoolParser(),
        )
        response = relevancy_link.run({"text": text, "topic": topic})

        # Return check
        relevancy_check = response["is_relevant"]
        logger.debug(f"Relevancy check response: {relevancy_check}")

        return relevancy_check

    @retry(max_retries=3)
    def remove_text_references(
        self, text: str, reference_flags: list | None = None
    ) -> str:
        """
        Check if the given text contains any of the reference flags, and clean them up
        :param text: The text to check
        :param reference_flags: The list of reference flags to check against
        """

        # Set default reference flags
        reference_flags = (
            reference_flags if reference_flags else DEFAULT_REFERENCE_FLAGS
        )

        # If the text does not contain any of the reference flags, return the text as is
        if not any(flag in text.lower() for flag in reference_flags):
            return text

        # Define template
        rewrite_template = MessagesTemplate(
            source=Messages(
                system="""You are a helpful editor. Your job is to make sure that text you are given
                 does not contain any references to a text or passage, since that is something that
                 the reader cannot see. The answer must stand on its own. If it already does, you should
                    reply with the same text. Otherwise, you should remove any references to the text or passage.
                    Reply with the edited or original text, delimited by triple backticks. Make sure to use proper spelling and grammar.

                 EXAMPLE:

                 Input: According to the text, the sky is blue because of Rayleigh scattering, which is the scattering of light by air molecules.
                 Output: The sky is blue because of Rayleigh scattering, which is the scattering of light by air molecules.

                 Input: The key reason given in the text for not using unit tests is that they are time-consuming.
                 Output: The key reason for not using unit tests is that they are time-consuming.
                 """,
                messages=[
                    Message(
                        role="user",
                        content="""Input text:```{text}```
                                """,
                    ),
                    Message(
                        role="assistant",
                        content="Revised text: ```",
                    ),
                ],
            )
        )
        logger.debug(
            f"Checking text <{get_first_n_words(text=text, n=10)}...> for references <{reference_flags}>"
        )
        rewrite_link = Link(
            template=rewrite_template,
            model=self.model,
            name="text_cleaner",
            output_key="cleaned_text",
            parser=TripleBacktickParser(prefill="```"),
        )
        response = rewrite_link.run({"text": text})

        # Return cleaned text
        cleaned_text = response["cleaned_text"]
        logger.debug(f"Cleaned text response: {cleaned_text}")

        return cleaned_text


if __name__ == "__main__":

    llm_helper = LLMHelper()
    text = "According to the text, the sky is blue because of Rayleigh scattering, which is the scattering of light by air molecules."
    topic = "The sky is blue"
    print(llm_helper.check_relevancy(text, topic))
