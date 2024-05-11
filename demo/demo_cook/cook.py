import logging
import sys

sys.path.extend(
    ["/Users/wouter.durnez/Documents/Repositories/test-ai-message-composer"]
)

from data.snippets import snippet
from fence.demo.demo_cook.utils import build_links, validate_recipe
from fence.chains import LinearChain
from fence.models import ClaudeInstantLLM
from fence.utils.logging import setup_logging

claude_model = ClaudeInstantLLM(source="test-ai-message-composer")

# Set up logging
setup_logging(log_level="INFO")
logger = logging.getLogger(__name__)


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

    # Validate recipe
    validate_recipe(recipe)

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
    # Some example recipes
    recipe = dict()

    recipe |= {
        "flavor": "sarcastic",
        "spelling": True,
        "verbosity": "shorter",
        "policy": [
            "Commitment: Do not commit to anything. Do not use any words that could be interpreted as a commitment.",
            "Branding: Make sure to mention the company name - Showpad - at least once. Do not overdo it. Do not "
            "mention any other company names.",
        ],
    }

    # Call handler
    response = handler(
        event={
            "recipe": recipe,
            "input": snippet,
        },
        context=None,
    )

    # Print response
    print(response["body"]["output"])
