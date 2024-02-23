from fence.demo.demo_search.prompt_templates import (
    COMBINE_TEMPLATE,
    SEARCH_CHUNK_TEMPLATE,
)
from fence.src.llm.links import Link
from fence.src.llm.models import LLM
from fence.src.llm.parsers import TripleBacktickParser
from fence.src.llm.templates import PromptTemplate


def build_links(llm: LLM):
    """
    Build links for the FAQ chain
    :param llm: LLM model
    :return: list of links
    """

    search_chunk_link = Link(
        name="search_chunk_link",
        template=PromptTemplate(
            template=SEARCH_CHUNK_TEMPLATE, input_variables=["question", "text"]
        ),
        llm=llm,
        parser=TripleBacktickParser(),
        output_key="search_chunk_output",
    )

    combine_link = Link(
        name="combine_link",
        template=PromptTemplate(
            template=COMBINE_TEMPLATE, input_variables=["search_chunk_outputs"]
        ),
        llm=llm,
        parser=TripleBacktickParser(),
        output_key="combine_output",
    )

    return {"search_chunk": search_chunk_link, "combine": combine_link}
