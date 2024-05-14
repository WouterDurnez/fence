import sys

sys.path.extend(
    ["/Users/wouter.durnez/Documents/Repositories/test-ai-message-composer"]
)

from fence.demo.demo_policy.utils import build_links
from fence.demo.demo_policy.presets import presets, test_cases
from fence.chains import LinearChain
from fence.llm import ClaudeHaiku
from fence.utils.logger import setup_logging

# claude_model = ClaudeInstantLLM(source="test-ai-message-composer")
claude_model = ClaudeHaiku(source="test-ai-message-composer", region="us-east-1")

# Set up logging
logger = setup_logging(log_level="INFO")


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

    # Build links and chain
    links = build_links(recipe, claude_model)
    chain = LinearChain(links=links, llm=claude_model)

    # Run chain
    response = chain.run(input_dict={"state": input_text, "recipe": recipe})

    # Build response dict
    response_dict = {
        "input": input_text,
        "reviewed": response,
        "output": response["state"],
        "input_words": len(input_text.split()),
        "output_words": len(response["state"].split()),
    }

    # Build response
    return {"statusCode": 200, "body": response_dict}


if __name__ == "__main__":

    # Set some example snippets
    snippet = "Last night, a grand gala was held in New York, organized by a renowned charity. Lavish decorations adorned the venue, and guests were served exquisite cuisine. Throughout the evening, speeches were given, and substantial donations were made. Memories were made, and the gala's success was celebrated by all."

    # Some example recipes
    recipe = dict()

    recipe |= {"policy": presets}

    # Loop through test cases
    for test_case in test_cases:

        logger.info(f"🧪 Running test case: {test_case}")

        # Get snippet
        snippet = test_case["text"]

        # Call handler
        response = handler(
            event={
                "recipe": recipe,
                "input": snippet,
            },
            context=None,
        )

        # Add response to test case
        test_case["transformed"] = response["body"]["output"]
