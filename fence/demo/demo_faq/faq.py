import logging
from pathlib import Path

from PyPDF2 import PdfReader

from fence import ClaudeInstantLLM
from fence.demo.demo_faq.utils import TextChunker, build_links
from fence.src.utils.base import parallelize, retry, setup_logging, time_it

setup_logging(log_level="DEBUG")

logger = logging.getLogger(__name__)


claude_model = ClaudeInstantLLM(source="test-faq")

NUMBER_OF_QUESTIONS = 3
ADD_SUMMARY = True
ADD_DESCRIPTION = True
ADD_TAGS = True
CHUNK_SIZE = 8_500
TRUNCATION_LIMIT = 30_000
MAX_RETRIES = 3
NUMBER_OF_WORKERS = 4


@time_it(threshold=0)
def handler(event, context):
    logger.info("👋 Let's rock!")

    # Parse event
    input_text = event.get("input_text", {})
    logger.debug(f"📥 Received input: {input_text}")
    logger.info(f"📥 Input text length: {len(input_text)} characters")

    # Truncate input text if it exceeds the truncation limit
    if len(input_text) > TRUNCATION_LIMIT:
        logger.warning(f"🔪 Truncating input text to {TRUNCATION_LIMIT} characters")
        input_text = input_text[:TRUNCATION_LIMIT]

    # Chunk the input text
    logger.info("🔪 Chunking input text...")
    chunker = TextChunker(text=input_text, chunk_size=CHUNK_SIZE, overlap=0.0)
    chunks = chunker.split_text()

    logger.info(
        f"🔪 Chunked input text into {len(chunks)} chunks of approximately {CHUNK_SIZE} characters each"
    )

    # Create chain
    links = build_links(llm=claude_model)

    @parallelize(max_workers=NUMBER_OF_WORKERS)
    @retry(max_retries=MAX_RETRIES)
    def run_chain(index: int, chunk: str):
        logger.info(f"🔗 Running chain for chunk {index}")

        # Run the chain
        faqs = links["faq"].run(
            input_dict={"state": chunk, "number_of_questions": NUMBER_OF_QUESTIONS}
        )["state"]
        logger.info(
            f"🔗 Chain for chunk {index} completed -- added {len(faqs['qa_pairs'])} QA pairs"
        )

        return faqs

    faq_results = run_chain(enumerate(chunks))

    # Extract output from chain
    summaries = [result["summary"] for result in faq_results]
    question_answers = [qa for result in faq_results for qa in result["qa_pairs"]]

    # If a summary questions is requested, add it to the list of questions
    meta_summary = None
    if ADD_SUMMARY:
        # Get a meta summary
        meta_summary = links["summary"].run(input_dict={"summaries": summaries})[
            "state"
        ]
        question_answers.insert(
            0,
            {
                "question": "What is the essence of this document?",
                "answer": meta_summary,
            },
        )

    # If tags are requested, generate them
    tags = None
    if ADD_TAGS:
        tags = links["tags"].run(input_dict={"summaries": summaries})["state"]

    # If a description question is requested, initialize the description
    description = None
    if ADD_DESCRIPTION:
        # Get a description
        description = links["description"].run(
            input_dict={
                "summaries": summaries,
                "extension": "pdf",
                "file_type": "document",
            }
        )["state"]

    # Remove context from question_answers
    question_answers = [
        {k: v for k, v in qa.items() if k != "context"} for qa in question_answers
    ]

    # Log output
    logger.info(
        f"📤 Out of {len(chunks)} chunks, {len(faq_results)} were successfully processed"
    )
    logger.info(f"📤 Returning {len(question_answers)} question-answers pairs")

    # Build response
    return {
        "statusCode": 200,
        "body": {
            "summary": meta_summary,
            "summaries": summaries,
            "tags": tags,
            "question_answers": question_answers,
            "description": description,
        },
    }


if __name__ == "__main__":
    # Set the topic
    topics = [
        "mma",
    ]

    # Responses
    responses = {}

    for TOPIC in topics[:]:
        # Load pdf file content
        file_path = Path(__file__).parent.parent.parent / "data" / f"{TOPIC}.pdf"
        reader = PdfReader(file_path)

        # Extract all text from all pages
        text = "\n".join([page.extract_text() for page in reader.pages])

        # Call the handler
        response = handler({"input_text": text}, None)

        # Store the response
        responses[TOPIC] = response

        # # Number of questions
        # print(f"Number of questions: {len(response['body']['question_answers'])}")
        #
        # print("Summary:")
        # print(response["body"]["summary"])
        #
        # print("QA pairs:")
        # pprint(response["body"]["question_answers"])
