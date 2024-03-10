import logging
import sys
from datetime import datetime

from dotenv import load_dotenv

sys.path.extend(
    ["/Users/wouter.durnez/Documents/Repositories/showpad_personal/fence"]
)

from fence.data.snippets import snippet
from fence.demo.demo_label.utils import build_links, FilenameProcessor
from fence.src.llm.chains import LinearChain
from fence.src.llm.models import ClaudeInstantLLM
from fence.src.utils.base import setup_logging

claude_model = ClaudeInstantLLM(source="test-ai-message-composer")

# Set up logging
setup_logging(log_level="INFO")
logger = logging.getLogger(__name__)

load_dotenv()

def handler(event: dict, context: any) -> dict:
    """
    Handler for the demo_cook lambda.
    """

    logger.info("👋 Let's rock!")

    # Parse event
    recipe = event.get("recipe", {})
    input_text = event.get("input", "")

    logger.info(f"📥 Received input: {input_text}")
    logger.info(f"📥 Received recipe: {recipe}")

    # Get info from recipe
    date = recipe.get("date", {}).get("value", None)
    date_format = recipe.get("date", {}).get("format", None)
    date_position = recipe.get("date", {}).get("location", "prefix")
    capitalisation = recipe.get("capitalisation", None)
    separator = recipe.get("separator", None)
    truncation_limit = recipe.get("truncation_limit", 50)
    remove_special_characters = recipe.get("remove_special_characters", False)
    policy = recipe.get("policy", [])

    logger.info(f"📥 Received date: {date}")
    logger.info(f"📥 Received capitalisation: {capitalisation}")
    logger.info(f"📥 Received separator: {separator}")
    logger.info(f"📥 Received truncation_limit: {truncation_limit}")
    logger.info(f"📥 Received remove_special_characters: {remove_special_characters}")

    # If date is not None, add a custom policy
    if date:
        policy.append(f"No dates: Strip the filename of any date information. Do not change ANYTHING else, just remove the date format.")

    # If there are no policies, skip the LLM interaction
    processed_filename = None
    if policy:

        # Build links and chain
        links = build_links(claude_model)
        chain = LinearChain(links=links, llm=claude_model)

        # Run chain
        processed_filename = chain.run(input_dict={"state": input_text, "recipe": recipe})['state']

    # Initialize FilenameProcessor
    if any([capitalisation, separator, remove_special_characters, date]):

        # Format date using the date format, input format is always YYYY-MM-DD
        if date:

            # Format date
            date = datetime.strptime(date, "%Y-%m-%d").strftime(date_format)


        filename_processor = FilenameProcessor(
            capitalisation=capitalisation,
            separator=separator,
            remove_special_characters=remove_special_characters,
            date=date,
            date_location=date_position,
            truncation_limit=truncation_limit,
        )

        # Process filename
        processed_filename = filename_processor.process_filename(processed_filename)

    # Build response dict
    response_dict = {
        "input": input_text,
        "output": processed_filename,
    }

    # Build response
    return {"statusCode": 200, "body": response_dict}


if __name__ == "__main__":

    filenames = [
        "this_is_a_filename_with_a_date_2021-10-01",
        "example.SomeReallyLongFILENAME also-Separators/OrCamelCase_orBothANDNumbers1234567890.did-i-mention-it-was-long",
        "someNewFiles_DK",
        "anotherFile.v1",
        "file1",
        "file2.loadsofnumbers1234567890",
        "filen@#ame_with_sp*cial_characters!@#$%^&*()_+",
        "a_screenshot_of_something_idk"
    ]

    # Some example recipes
    recipe = dict()

    recipe |= {
        "date":
            {
                'value': '2024-10-01',
                'format': '%Y%m%d',
                'location': 'suffix',
            },
        "capitalisation": "title",
        "separator": "_",
        "truncation_limit": 50,
        "remove_special_characters": True,
        "policy": ["Do not use the word 'screenshot', but use 'image' instead."],
    }

    # Call handler for each filename
    for filename in filenames:
        response = handler(
            event={
                "recipe": recipe,
                "input": filename,
            },
            context=None,
        )

        # Print response
        logger.critical(f"Recevied input: {filename} \t--> Output: {response['body']['output']}")

