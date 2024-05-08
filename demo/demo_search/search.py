import os

from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.documents import Document

from fence.demo.demo_search.test_questions import *
from fence.demo.demo_search.utils import build_links
from fence.models import ClaudeInstantLLM
from fence.utils.base import DATA_DIR, parallelize, retry, setup_logging, time_it

claude_model = ClaudeInstantLLM(source="test-search")
embeddings = BedrockEmbeddings()

# Set up logging
logger = setup_logging(os.environ.get("LOG_LEVEL", "INFO"))

NUMBER_OF_WORKERS = 5
MAX_RETRIES = 3

VECTOR_DB_PATH = DATA_DIR / "search" / "random_db"


@time_it(threshold=0)
def handler(event: dict, context: any) -> dict:
    """
    Handler for the demo_cook lambda.
    """

    logger.info("👋 Let's rock!")
    question = event.get("input", "")

    # Retrieve relevant documents
    docsearch = Chroma(
        persist_directory=str(VECTOR_DB_PATH), embedding_function=embeddings
    )
    relevant_documents = docsearch.similarity_search(query=question, k=5)

    # Call model
    links = build_links(llm=claude_model)

    @parallelize(max_workers=NUMBER_OF_WORKERS)
    @retry(max_retries=MAX_RETRIES)
    def run_chain(index: int, document: Document):
        logger.info(f"🔗 Running chain for chunk {index}")

        # Run the chain
        search_result = links["search_chunk"].run(
            input_dict={"question": question, "text": document.page_content}
        )["search_chunk_output"]
        logger.info(f"🔗 Chain for chunk {index} completed.")

        if "<not in text>" in search_result:
            return

        return {"result": search_result, "source": document.metadata["source"]}

    search_results = run_chain(relevant_documents)

    # Filter out None values
    search_results = [result for result in search_results if result is not None]

    # Get sources
    sources = {result["source"] for result in search_results}

    # Get answers
    answers = [result["result"] for result in search_results]

    # If all answers are <not in text>, return
    if all(answer == "<not in text>" for answer in answers):
        return {
            "input": question,
            "documents": relevant_documents,
            "search_result": search_results,
            "combine_output": "No relevant information found in the documents.",
            "sources": None,
        }

    # Combine the results
    combine_output = links["combine"].run(input_dict={"search_chunk_outputs": answers})[
        "combine_output"
    ]

    return {
        "input": question,
        "documents": relevant_documents,
        "search_result": search_results,
        "combine_output": combine_output,
        "sources": sources,
    }


if __name__ == "__main__":
    responses = []
    questions = [
        "What is the traveling salesman problem?",
        "What is MMA?",
        "What is the capital of France?",
        "What is Tropical Storm Brenda?",
        "What is the shape of the earth?",
    ]

    for question in questions:
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
