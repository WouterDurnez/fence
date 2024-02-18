import logging
from math import ceil

from fence import LLM, Link, PromptTemplate
from fence.demo_faq.prompt_templates import (
    DESCRIPTION_TEMPLATE,
    FAQ_TEMPLATE,
    SUMMARY_TEMPLATE,
)
from fence.src.llm.parsers import TOMLParser, TripleBacktickParser
from fence.src.utils.base import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class TextChunker:
    def __init__(self, text: str, chunk_size: int, overlap: float | None = 0.05):
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

    def split_text(self) -> list[str]:
        core_chunk_size = self._get_approximate_chunk_size(
            text=self.text, chunk_size=int(self.chunk_size / (1 + self.overlap))
        )

        chunk_size_with_overlap = (
            (core_chunk_size + int(core_chunk_size * self.overlap))
            if self.overlap
            else core_chunk_size
        )

        logger.debug(
            f"Chunking {len(self.text)} characters into {self.chunk_size=}-sized chunks with : {core_chunk_size=}, {chunk_size_with_overlap=}"
        )

        remaining_text = self.text
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


def build_links(llm: LLM):
    """
    Build links for the FAQ chain
    :param llm: LLM model
    :return: list of links
    """

    faq_link = Link(
        name="faq_link",
        template=PromptTemplate(template=FAQ_TEMPLATE, input_variables=["state"]),
        llm=llm,
        parser=TOMLParser(),
        output_key="faq_output",
    )

    summary_link = Link(
        name="summary_link",
        template=PromptTemplate(
            template=SUMMARY_TEMPLATE, input_variables=["summaries"]
        ),
        llm=llm,
        parser=TripleBacktickParser(),
        output_key="summary_output",
    )

    description_link = Link(
        name="description_link",
        template=PromptTemplate(
            template=DESCRIPTION_TEMPLATE,
            input_variables=["summaries", "extension", "file_type"],
        ),
        llm=llm,
        parser=TripleBacktickParser(),
        output_key="description_output",
    )

    return {"faq": faq_link, "summary": summary_link, "description": description_link}
