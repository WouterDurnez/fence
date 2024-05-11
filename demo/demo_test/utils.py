import json
import logging
import re
from copy import deepcopy

from fence import (
    LLM,
    Chain,
    ClaudeInstant,
    Link,
    StringTemplate,
    TransformationLink,
)
from demo.demo_test.prompt_templates import TEST_TEMPLATE, VERIFICATION_TEMPLATE
from fence.parsers import TOMLParser
from fence.utils.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

claude_instant_model = ClaudeInstant(source="test", temperature=0.5)


def clean_question(input_dict: dict) -> dict:
    """
    Clean the question:
    - remove the "According to the passage, " prefix and capitalize the first letter of the question.
    - capitalize the responses
    - remove the reasoning key from the responses
    :param input_dict: the input dictionary containing the full question
    """
    # Copy the data
    question = deepcopy(input_dict["full_question"])

    # Convert the data to a dictionary if it is a string
    if isinstance(question, str):
        question = json.loads(question)

    # Question's will start with "According to the passage, "
    # Remove this from the question, and capitalize the first letter
    parsed = question["question"]
    parsed = parsed.replace("According to the passage, ", "")
    parsed = (parsed[0].upper() + parsed[1:]) if len(parsed) > 0 else parsed
    question["question"] = parsed

    # Clean up responses
    for response in question["responses"]:
        # Responses should be capitalized
        response["text"] = response["text"].capitalize()

        # Remove the reasoning key: that was just to get the LLM to think a little harder
        response.pop("reason", None)

    return question


def strip_question(input_dict: dict) -> dict:
    """
    Strip the correctness from the responses

    Example:
    Input:
    {
        'question': 'What is the capital of France?',
        'responses': [
            {'text': 'Paris', 'correct': True},
            {'text': 'London', 'correct': False},
            {'text': 'Berlin', 'correct': False}
        ]
    }
    Output:
    {
        'question': 'What is the capital of France?',
        'responses': [
            {'text': 'Paris'},
            {'text': 'London'},
            {'text': 'Berlin'}
        ]
    }

    :param question: the question to strip
    :return: the stripped question
    """

    # Copy the data
    question = deepcopy(input_dict["clean_question"])

    # Convert the data to a dictionary if it is a string
    if isinstance(question, str):
        question = json.loads(question)

    # Convert the data to a dictionary if it is a string
    if isinstance(question, str):
        question = json.loads(question)

    # Remove the 'correct' key from the responses
    if "responses" in question and isinstance(question["responses"], list):
        for response in question["responses"]:
            response.pop("correct", None)

    return question


def verify_answer(input_dict: dict) -> dict:
    """
    Verify the answer. Returns True if the answer is correct, False otherwise.

    :param inputs: input dict containing the question and the answer
    :return: True if the answer is correct, False otherwise
    """

    answer = input_dict["answered_question"].strip()

    # Try extracting the selected answer index
    try:
        selected_answer_idx = int(answer)
    except ValueError:
        logger.error("Error parsing selected answer index. Trying to find integer.")
        try:
            selected_answer_idx = int(re.findall(r"\d+", answer)[0])
        except ValueError:
            logger.error("Error parsing selected answer index. Returning None.")
            return None

    question = input_dict["full_question"]

    # Get the correct answer
    correct_answer_idx = None
    for idx, response in enumerate(question["responses"]):
        if response["correct"]:
            correct_answer_idx = idx
            break

    # Print correct and selected answer if they are not the same
    if not selected_answer_idx == correct_answer_idx:
        return False
    else:
        return True


def build_links(llm: LLM):
    """
    Build links for the FAQ chain
    :param llm: LLM model
    :return: list of links
    """

    test_link = Link(
        name="test_link",
        template=StringTemplate(source=TEST_TEMPLATE),
        llm=llm,
        parser=TOMLParser(),
        output_key="full_question",
    )

    clean_link = TransformationLink(
        name="clean_link",
        function=clean_question,
        input_keys=["full_question"],
        output_key="clean_question",
    )

    strip_link = TransformationLink(
        name="strip_link",
        function=strip_question,
        input_keys=["clean_question"],
        output_key="question_stripped",
    )

    verification_link = Link(
        name="verification_link",
        template=StringTemplate(source=VERIFICATION_TEMPLATE),
        llm=llm,
        parser=None,
        output_key="answered_question",
    )

    check_link = TransformationLink(
        name="check_link",
        function=verify_answer,
        input_keys=["full_question", "answered_question"],
        output_key="verification",
    )

    return [test_link, clean_link, strip_link, verification_link, check_link]


if __name__ == "__main__":
    # Build test chain
    links = build_links(llm=claude_instant_model)
    chain = Chain(links=links)
