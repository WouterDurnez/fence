from demo.demo_search.prompt_templates import (
    COMBINE_TEMPLATE,
    SEARCH_CHUNK_TEMPLATE,
)
from fence.links import Link
from fence.models import LLM
from fence.parsers import TripleBacktickParser
from fence.templates import StringTemplate
from fence.utils.logger import setup_logging

import uuid

# Set up logging
logger = setup_logging(__name__)


def format_result_list(search_results: list):
    """
    Given a list of search results, format them into a string for the LLM:

    :param search_results: list of search results (strings)
    """

    return "\n".join([f" - {result}" for result in search_results])


def build_links(llm: LLM):
    """
    Build links for the FAQ chain
    :param llm: LLM model
    :return: list of links
    """

    search_chunk_link = Link(
        name="search_chunk_link",
        template=StringTemplate(
            source=SEARCH_CHUNK_TEMPLATE,
        ),
        llm=llm,
        parser=TripleBacktickParser(),
        output_key="search_chunk_output",
    )

    combine_link = Link(
        name="combine_link",
        template=StringTemplate(
            source=COMBINE_TEMPLATE,
        ),
        llm=llm,
        parser=TripleBacktickParser(),
        output_key="combine_output",
    )

    return {"search_chunk": search_chunk_link, "combine": combine_link}


def build_response(search_results: list, answer: str):
    """
    Build the response for the AI powered search lambda

    :param search_results: list of search results, containing `document` and `result`
    :param combined_answer: combined answer
    """

    # Get unique documents from the search results (can't use set because Document is not hashable)
    sources = set(

            result["document"].metadata["source"]

        for result in search_results
    )


    # Build the response
    answers = [
        {
            "text": answer,
            "sources": sources,
            "id": str(uuid.uuid4()),
        }
    ]

    return answers
