import os

from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.documents import Document

from demo.demo_search.test_questions import *
from demo.demo_search.utils import build_links, format_result_list, \
    build_response
from fence.models import ClaudeHaiku
from fence.utils.base import DATA_DIR, time_it
from fence.utils.optim import retry, parallelize
from fence.utils.logger import setup_logging

claude_model = ClaudeHaiku(source="test-search")
embeddings = BedrockEmbeddings()

# Set up logging
logger = setup_logging(__name__, log_level='WARNING', serious_mode=False)

NUMBER_OF_WORKERS = 5
MAX_RETRIES = 3

VECTOR_DB_PATH = DATA_DIR / "search" / "paper_db"


@time_it(threshold=0)
def handler(event: dict, context: any) -> dict:
    """
    Handler for the demo_cook lambda.
    """

    logger.info("👋 Let's rock!")
    query = event.get("input", "")

    # Retrieve relevant documents
    docsearch = Chroma(
        persist_directory=str(VECTOR_DB_PATH), embedding_function=embeddings
    )
    docs = docsearch.similarity_search(query=question, k=5)
    logger.info(f"🔍 Found {len(docs)} relevant documents.")
    print(docs)

    # Initialize the LLM links
    links = build_links(llm=claude_model)

    @parallelize(max_workers=NUMBER_OF_WORKERS)
    @retry(max_retries=MAX_RETRIES)
    def run_chain(document: Document):
        logger.info("🔗 Running chain for chunk")

        # Run the chain
        search_result = links["search_chunk"].run(
            input_dict={"question": query, "text": document.page_content}
        )["search_chunk_output"]
        logger.info("🔗 Chain for chunk completed.")

        if "<not in text>" in search_result:
            return

        return {"result": search_result, "document": document}

    # Run the chain in parallel
    search_results = run_chain(docs)

    # Filter out None values
    search_results = [result for result in search_results if
                      result is not None]
    logger.info(f"Search results: {search_results}")

    # Get answers
    answers = [result["result"] for result in search_results]

    # If all answers are <not in text>
    if all(answer == "<not in text>" for answer in answers):
        logger.info("No valid sources found, all answers are <not in text>")
        return {
            "input": question,
            "documents": docs,
            "search_result": search_results,
            "combine_output": "<not in text>",
            "sources": None,
        }

    # Combine the results: generate a single answer from the multiple answers, or return <contradictory>
    combined_answer = links["combine"].run(
        input_dict={
            "search_results": format_result_list(search_results=answers)}
    )["combine_output"]

    # If the output is <contradictory>, return early
    logger.info(f"Combined answer: {combined_answer}")
    if combined_answer == "<contradictory>":
        logger.info("Contradictory answers found")
        return {
            "input": question,
            "documents": docs,
            "search_result": search_results,
            "combine_output": "<contradictory>",
            "sources": None,
        }

    # Build the response
    answers = build_response(search_results=search_results,
                             answer=combined_answer)
    sources = list(set(answers[0]["sources"]))

    return {
        "input": question,
        "documents": docs,
        "search_result": search_results,
        "combine_output": combined_answer,
        "sources": sources,
    }


if __name__ == "__main__":
    responses = []
    questions = [
        # "What is the traveling salesman problem?",
        # "What is MMA?",
        # "What is the capital of France?",
        # "What is Tropical Storm Brenda?",
        "What is the shape of the earth?",
    ]

    for question in questions[:]:
        logger.critical(f"Question: {question}")
        response = handler(
            event={
                "input": question,
            },
            context=None,
        )
        logger.critical(f"Answer: {response['combine_output']}")
        logger.critical(
            f"Sources [{len(response['sources']) if response['sources'] is not None else 0}]: {response['sources']}"
        )
        responses.append(response)

    # Question, answer and sources only
    responses_filtered = [
        {
            "question": response["input"],
            "answer": response["combine_output"],
            "sources": response["sources"],
        }
        for response in responses
    ]
