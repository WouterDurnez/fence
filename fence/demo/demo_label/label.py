import logging
import os
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
logger = setup_logging(log_level=os.environ.get("LOG_LEVEL", "INFO"))
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
        policy.append(f"No dates: Strip the filename of any date information. Do not change ANYTHING else, just remove the date format."
                      f""
                      f"Example 1: 'filename_2021-10-01' -> 'filename'"
                      f"Example 2: '2021-10-01_filename' -> 'filename'"
                      f"Example 3: 'something.v1.march24' -> 'something.v1'")

    # If there are no policies, skip the LLM interaction
    processed_filename = None
    if policy:

        # Build links and chain
        links = build_links(claude_model)
        chain = LinearChain(links=links, llm=claude_model)

        # Run chain
        processed_filename = chain.run(input_dict={"state": input_text, "recipe": recipe})['state']

        # Copy for output
        llm_output = processed_filename

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
        "llm_output": llm_output,
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
        "filen@me_with_sp*cial_characters!@#$%^&*()_+",
        "a_screenshot_of_something_idk",
        "my_file_april-1st",
        "20241001_Valid_Filename_Idempotence_Check",
    ]

    expected_outputs = [
        "20241001_This_Is_A_Filename",
        "20241001_Example_Some_Really_Long_Filenamealso_Sep",
        "20241001_Some_New_Files_Dk",
        "20241001_Another_File_V1",
        "20241001_File1",
        "20241001_File2_Loadsofnumbers1234567890",
        "20241001_Filen_Me_With_Sp_Cial_Characters",
        "20241001_A_Image_Of_Something_Idk",
        "20241001_My_File",
        "20241001_Valid_Filename_Idempotence_Check",
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
        "truncation_limit": 20,
        "remove_special_characters": True,
        "policy": ["Do not use the word 'screenshot', but use 'image' instead. Do not change anything else."],
    }

    # Call handler for each filename
    for filename, expected in zip(filenames, expected_outputs):
        response = handler(
            event={
                "recipe": recipe,
                "input": filename,
            },
            context=None,
        )

        # Print response
        logger.critical(f"Received input: \t{filename}"
                        f"\nLLM output: \t\t{response['body']['llm_output']}"
                        f"\nTransformed output: \t{response['body']['output']}"
                        f"\nExpected output: \t{expected}"
                        f"\nOutput length: \t\t{len(response['body']['output'])} characters"
                        f"\nCorrect? \t\t{response['body']['output'] == expected}"
                        f"\n")

