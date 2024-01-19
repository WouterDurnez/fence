import logging
from pprint import pprint
from typing import Dict

from data.snippets import snippet_drunk, snippet
from lambdas.cook.lib.llm.base_templates import BASE_TEMPLATE, FLAVOR_TEMPLATE, POLICY_TEMPLATE, VERBOSITY_TEMPLATE, SPELLING_TEMPLATE
from lambdas.cook.lib.llm.chains import Link, LinearChain
from lambdas.cook.lib.llm.models import ClaudeInstantLLM, LLM
from lambdas.cook.lib.llm.parsers import TOMLParser, TripleBacktickParser
from lambdas.cook.lib.llm.templates import PromptTemplate
from lambdas.cook.lib.utils.base import setup_logging, validate_recipe

claude_model = ClaudeInstantLLM(source="test-ai-message-composer")

# Set up logging
setup_logging(log_level='INFO')
logger = logging.getLogger(__name__)

# Define link templates and parsers
LINK_TEMPLATES = {
    "spelling": SPELLING_TEMPLATE,
    "policy": POLICY_TEMPLATE,
    "flavor": FLAVOR_TEMPLATE,
    "verbosity": VERBOSITY_TEMPLATE,
}
LINK_PARSERS = {
    "spelling": TOMLParser(),
    "policy": TripleBacktickParser(),
    "flavor": TripleBacktickParser(),
    "verbosity": TripleBacktickParser(),
}


def build_links(recipe: dict, llm: LLM) -> list[Link]:
    """
    Build and return a list of links based on the provided recipe and templates.
    :param recipe: recipe JSON object
    :param llm: LLM instance
    """
    links = []

    for key in ["verbosity", "policy", "flavor", "spelling"]:
        if key in recipe:
            link = Link(
                name=f"{key}_link",
                template=PromptTemplate(template=LINK_TEMPLATES[key], input_variables=["state", "recipe"]),
                llm=llm,
                parser=LINK_PARSERS[key],
                output_key=f"{key}_output",
            )
            links.append(link)

    return links

def handler(event: Dict, context: any) -> Dict:
    """
    Handler for the cook lambda.
    """

    # Parse event
    recipe = event.get("recipe", {})
    input_text = event.get("input", "")

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
        'spelling': True,
        "verbosity": "longer",
        "policy": ["Commitment: Do not commit to anything. Do not use any words that could be interpreted as a commitment.",
                     "Branding: Make sure to mention the company name - Showpad - at least once. Do not overdo it. Do not mention any other company names."]
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
    print(response['body']['output'])