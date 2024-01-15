from pprint import pprint
from tomllib import TOMLDecodeError

from data.snippets import snippet
from lambdas.cook.lib.base_templates import base_template
from lambdas.cook.lib.llm.models import ClaudeInstantLLM
from lambdas.cook.lib.llm.templates import PromptTemplate
from lambdas.cook.lib.utils.base import parse_toml

claude_model = ClaudeInstantLLM(source="test-ai-message-composer")


def handler(event, context):
    # Parse event
    recipe = event["recipe"]
    input_text = event["input"]

    # Build prompt
    template = PromptTemplate(
        template=base_template, input_variables=["input", "recipe"]
    )
    prompt = template.render(input=input_text, recipe=recipe)

    # Invoke model
    response = claude_model(prompt=prompt)

    # Parse response
    try:
        response_dict = parse_toml(response)
    except TOMLDecodeError as e:
        return {
            "statusCode": 500,
            "body": {
                "error": "Failed to parse TOML from LLM response",
                "message": str(e),
                "response": response,
            },
        }

    # Calculate some metrics
    response_dict["input_token_words"] = len(response_dict["input"].split())
    response_dict["reviewed_token_words"] = len(response_dict["reviewed"].split())

    # Build response
    response = {
        "statusCode": 200,
        "body": response_dict,
    }

    return response


if __name__ == "__main__":
    # Some example recipes
    recipe = dict()
    recipe |= {
        "flavor": "severely depressed",
        #'spelling': True,
        # "verbosity": "shorter",
    }
    # recipe |= {
    #     "policy": "- Commitment: Do not commit to anything. Do not use any words that could be interpreted as a commitment.\n"
    #     "- Branding: Make sure to mention the company name - Showpad - at least once. Do not overdo it. Do not mention any other company names.\n"
    # }

    response = handler(
        {
            "recipe": recipe,
            "input": snippet,
        },
        None,
    )
    pprint(response)
