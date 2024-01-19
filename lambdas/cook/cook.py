import logging
from pprint import pprint

from data.snippets import snippet_drunk, snippet
from lambdas.cook.lib.llm.base_templates import BASE_TEMPLATE, FLAVOR_TEMPLATE, POLICY_TEMPLATE, VERBOSITY_TEMPLATE, SPELLING_TEMPLATE
from lambdas.cook.lib.llm.chains import Link, LinearChain
from lambdas.cook.lib.llm.models import ClaudeInstantLLM
from lambdas.cook.lib.llm.parsers import TOMLParser, TripleBacktickParser
from lambdas.cook.lib.llm.templates import PromptTemplate
from lambdas.cook.lib.utils.base import setup_logging, validate_recipe

claude_model = ClaudeInstantLLM(source="test-ai-message-composer")

# Set up logging
setup_logging(log_level='INFO')
logger = logging.getLogger(__name__)

def handler(event, context):
    """
    Handler for the cook lambda.
    """

    # Parse event
    recipe = event["recipe"]
    input_text = event["input"]

    # Validate recipe
    validate_recipe(recipe)

    # Keys in recipe
    keys = list(recipe.keys())

    # Potential keys are flavor, policy, verbosity, spelling
    # We want to order them in this way:
    # 1. verbosity
    # 2. policy
    # 3. flavor
    # 4. spelling

    keys = [key for key in ["verbosity", "policy", "flavor", "spelling"] if key in recipe]

    # Build links for each key
    spelling_link = Link(name='spelling_link', template=PromptTemplate(template=SPELLING_TEMPLATE, input_variables=["state", "recipe"]), llm=claude_model, parser=TOMLParser(), output_key="spelling_output")
    policy_link = Link(name='policy_link', template=PromptTemplate(template=POLICY_TEMPLATE, input_variables=["state", "recipe"]), llm=claude_model, parser=TripleBacktickParser(), output_key="policy_output")
    flavor_link = Link(name='flavor_link', template=PromptTemplate(template=FLAVOR_TEMPLATE, input_variables=["state", "recipe"]), llm=claude_model, parser=TripleBacktickParser(), output_key="flavor_output")
    verbosity_link = Link(name = 'verbosity_link', template=PromptTemplate(template=VERBOSITY_TEMPLATE, input_variables=["state", "recipe"]), llm=claude_model, parser=TripleBacktickParser(), output_key="verbosity_output")

    # Build chain of links, depending on keys
    link_map = {
        "spelling": spelling_link,
        "policy": policy_link,
        "flavor": flavor_link,
        "verbosity": verbosity_link,
    }

    links = [link_map[key] for key in keys]
    chain = LinearChain(links=links, llm=claude_model)

    # Run chain
    response = chain.run(input_dict={"state": input_text, "recipe": recipe})
    pprint(response)

    # Build response dict
    response_dict = {
        "input": input_text,
        "reviewed": response,
        "output": response["state"],
    }

    # Calculate some metrics
    response_dict["input_words"] = len(response_dict["input"].split())
    response_dict["output_words"] = len(response_dict["output"].split())

    # Build response
    response_dict = {
        "statusCode": 200,
        "body": response_dict,

    }

    return response_dict


if __name__ == "__main__":
    # Some example recipes
    recipe = dict()

    recipe |= {
        "flavor": "sarcastic",
        'spelling': True,
        "verbosity": "shorter",
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
    #pprint(response)
