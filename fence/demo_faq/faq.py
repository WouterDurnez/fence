import logging
import os
from math import ceil
from pathlib import Path
from pprint import pprint
from typing import Any, Dict, List, Optional
from PyPDF2 import PdfReader

from fence import ClaudeInstantLLM, LinearChain
from fence.demo_faq.utils import split_text, build_links

logger = logging.getLogger(__name__)


claude_model = ClaudeInstantLLM(source="test-faq")

NUMBER_OF_QUESTIONS = 2


def handler(event, context):
    logger.info("👋 Let's rock!")

    # Parse event
    input_text = event.get("input_text", {})
    logger.info(f"📥 Received input: {input_text}")

    # Chunk the input text
    logger.info("🔪 Chunking input text...")
    chunks = split_text(text=input_text, chunk_size=30000, overlap=0.1)

    logger.info(
        f"🔪 Chunked input text into {len(chunks)} chunks of approximately 10k characters each"
    )

    # Create chain
    links = build_links(llm=claude_model)

    # Run chain for each chunk
    faq_results = []
    for chunk in chunks[:]:
        logger.info(f"🔗 Running chain for chunk: {chunk}")
        faq_results.append(
            links["faq"].run(
                input_dict={"state": chunk, "number_of_questions": NUMBER_OF_QUESTIONS}
            )["state"]
        )

    # Extract output from chain
    summaries = [result["summary"] for result in faq_results]
    question_answers = [qa for result in faq_results for qa in result["qa_pairs"]]

    # Get a meta summary
    meta_summary = links["summary"].run(input_dict={"summaries": summaries})["state"]

    # Build response
    return {
        "statusCode": 200,
        "body": {
            "summary": meta_summary,
            "summaries": summaries,
            "question_answers": question_answers,
        },
    }


if __name__ == "__main__":
    # Load pdf file content
    file_path = Path(__file__).parent.parent / "data" / "mma.pdf"
    reader = PdfReader(file_path)

    # Extract all text from all pages
    text = "\n".join([page.extract_text() for page in reader.pages])

    # Call the handler
    response = handler({"input_text": text}, None)

    print("Summary:")
    print(response["body"]["summary"])

    print("QA pairs:")
    pprint(response["body"]["question_answers"])
