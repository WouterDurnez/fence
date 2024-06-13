import os
import logging
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
from fence.utils.nlp import LLMHelper


claude_model = ClaudeHaiku(source="test-search")
embeddings = BedrockEmbeddings()

# Set up logging
logger = setup_logging(__name__, log_level='INFO', serious_mode=False)

# Exclude boto3 from logging
logging.getLogger("boto3").setLevel(logging.CRITICAL)

NUMBER_OF_WORKERS = 5
MAX_RETRIES = 3
MAX_DOCUMENTS = 10
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
    docs = docsearch.similarity_search(query=query, k=MAX_DOCUMENTS)
    logger.info(f"🔍 Found {len(docs)} relevant documents.")

    # Initialize the LLM links
    links = build_links(llm=claude_model)

    @parallelize(max_workers=NUMBER_OF_WORKERS)
    @retry(max_retries=MAX_RETRIES)
    def run_chain(document: Document):
        logger.debug("🔗 Running chain for chunk")

        # Run the chain
        search_result = links["search_chunk"].run(
            input_dict={"question": query, "text": document.page_content}
        )["search_chunk_output"]
        logger.debug("🔗 Chain for chunk completed.")
        logger.info(f"Search result: {search_result}")

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
            "input": query,
            "documents": docs,
            "search_result": search_results,
            "combine_output": "<not in text>",
            "sources": None,
        }

    @retry(max_retries=MAX_RETRIES)
    def combine_answers(answers: list) -> str:
        """
        Combine the answers
        :param answers: List of answers
        :return: Combined answer
        """

        combined_answer = links["combine"].run(
            input_dict={
                "search_results": format_result_list(search_results=answers)}
        )["combine_output"]

        # Clean up the answer
        llm_helper = LLMHelper(model=claude_model)
        combined_answer = llm_helper.remove_text_references(combined_answer)

        return combined_answer

    final_answer = combine_answers(answers)

    # If the output is <contradictory>, return early
    logger.info(f"Combined answer: {final_answer}")
    if final_answer == "<contradictory>":
        logger.info("Contradictory answers found")
        return {
            "input": query,
            "documents": docs,
            "search_result": search_results,
            "combine_output": "<contradictory>",
            "sources": None,
        }



    # Build the response
    answers = build_response(search_results=search_results,
                             answer=final_answer)
    sources = list(set(answers[0]["sources"]))

    return {
        "input": query,
        "documents": docs,
        "search_result": search_results,
        "combine_output": final_answer,
        "sources": sources,
    }


if __name__ == "__main__":

    llm_helper = LLMHelper(model=claude_model)
    test = llm_helper.remove_text_references("The text mentions that 72 hours is a good rest period in between heavy workouts.")
    test2 = llm_helper.remove_text_references("The text mentions that 72 hours is a good rest period in between heavy workouts.", reference_flags=["text"])
    #
    #
    # responses = []
    # questions = [
    #     "What is the traveling salesman problem?",
    #     "What is MMA?",
    #     "What is the capital of France?",
    #     "What is Tropical Storm Brenda?",
    #     "What is the shape of the earth?",
    #     "What is AI?",
    #     "How does AI work?",
    #     "How does G-Eval work?",
    #     "What is low-rank approximation?",
    #     "How does Dungeons and Dragons work?",
    #     "How does low-rank approximation work?",
    #
    # ]
    #
    # for question in questions[:]:
    #     logger.critical(f"Question: {question}")
    #     response = handler(
    #         event={
    #             "input": question,
    #         },
    #         context=None,
    #     )
    #     logger.critical(f"Answer: {response['combine_output']}")
    #     logger.critical(
    #         f"Sources [{len(response['sources']) if response['sources'] is not None else 0}]: {response['sources']}"
    #     )
    #     responses.append(response)
    #
    # # Question, answer and sources only
    # responses_filtered = [
    #     {
    #         "question": response["input"],
    #         "answer": response["combine_output"],
    #         "sources": response["sources"],
    #     }
    #     for response in responses
    # ]
