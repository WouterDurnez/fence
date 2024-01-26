from math import ceil
import logging

from fence import LinearChain, LLM, Link, PromptTemplate
from fence.src.llm.parsers import TOMLParser, TripleBacktickParser
from fence.src.utils.base import setup_logging

from fence.demo_faq.prompt_templates import FAQ_TEMPLATE, SUMMARY_TEMPLATE

setup_logging()
logger = logging.getLogger(__name__)

def get_approximate_chunk_size(text: str, chunk_size: int) -> int:
    """
    Get approximate chunk size for a given text and chunk size
    :param text: text string
    :param chunk_size: desired chunk size target
    :return: recalculated chunk size
    """

    # Calculation: get (floored) number of chunks, then
    # divide text length by that number to get approximate chunk size
    approx_chunk_count = ceil(len(text) / chunk_size)
    approx_chunk_size = ceil(len(text) / approx_chunk_count)
    return approx_chunk_size


def split_text(text: str, chunk_size: int, overlap: float | None = 0.05) -> list:
    """
    Split text into chunks of approximately equal size, bearing chunk_size in mind as a target.
    Take overlap into account.
    :param text: text string to split
    :param chunk_size: target chunk size
    :param overlap: overlap between chunks as a percentage of chunk size
    :return: list of text chunks
    """

    # Calculate chunk size that best fits the target chunk size, targeting equal chunk sizes
    # Core chunk size + overlap should be as close to target chunk size as possible
    core_chunk_size = get_approximate_chunk_size(text=text, chunk_size=int(chunk_size / (1 + overlap)))

    # Add overlap to chunk size if overlap is specified
    chunk_size_with_overlap = (core_chunk_size + int(core_chunk_size * overlap)) if overlap else core_chunk_size

    logger.debug(
        f"Chunking {len(text)} characters into {chunk_size=}-sized chunks with : {core_chunk_size=}, {chunk_size_with_overlap=}"
    )

    # Process text in chunks
    remaining_text = text
    chunks = []

    # Keep going until we're out of text
    while len(remaining_text) > core_chunk_size:
        chunk = (
            remaining_text[:chunk_size_with_overlap]
            if len(remaining_text) > chunk_size_with_overlap
            else remaining_text
        )
        chunks.append(chunk)
        remaining_text = remaining_text[core_chunk_size:]

    # Add remaining text to chunks
    if remaining_text:
        chunks.append(remaining_text)

    logger.debug(
        f"Chunking complete. {len(chunks)} chunks created. Smallest chunk size: {min([len(chunk) for chunk in chunks])}"
    )

    return chunks



def build_links(llm:LLM):
    """
    Build links for the FAQ chain
    :param llm: LLM model
    :return: list of links
    """

    faq_link = Link(
        name="faq_link",
        template=PromptTemplate(
            template=FAQ_TEMPLATE, input_variables=["state"]
        ),
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

    return {
        'faq': faq_link,
        'summary': summary_link
    }