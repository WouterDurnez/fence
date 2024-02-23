from fence import LLM, Link, PromptTemplate
from fence.demo.demo_cook.prompt_templates import (
    FLAVOR_TEMPLATE,
    POLICY_TEMPLATE,
    SPELLING_TEMPLATE,
    VERBOSITY_TEMPLATE,
)
from fence.src.llm.parsers import TOMLParser, TripleBacktickParser

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

VALID_FLAVORS = ["sarcastic", "formal", "informal"]
VALID_VERBOSITY = ["shorter", "longer"]


def validate_recipe(recipe: dict):
    """
    Validate whether a recipe JSON object is valid.
    :param dict recipe: JSON object containing the recipe
    :return: True if the recipe is valid, raise ValueError otherwise
    """

    # Check if the recipe contains any invalid keys
    valid_keys = ["flavor", "verbosity", "spelling", "policy"]
    invalid_keys = [key for key in recipe.keys() if key not in valid_keys]
    if invalid_keys:
        raise ValueError(f"Invalid recipe keys: {invalid_keys}")

    # Check if the recipe contains any invalid values
    if "flavor" in recipe:
        if recipe["flavor"] not in VALID_FLAVORS:
            raise ValueError(f"Invalid flavor: {recipe['flavor']}")
    if "verbosity" in recipe:
        if recipe["verbosity"] not in VALID_VERBOSITY:
            raise ValueError(f"Invalid verbosity: {recipe['verbosity']}")
    if "spelling" in recipe:
        if not isinstance(recipe["spelling"], bool):
            raise ValueError(f"Invalid spelling: {recipe['spelling']}")
    if "policy" in recipe:
        if not (
            isinstance(recipe["policy"], list) or isinstance(recipe["policy"], str)
        ):
            raise ValueError(f"Invalid policy: {recipe['policy']}")
        if isinstance(recipe["policy"], str):
            recipe["policy"] = [recipe["policy"]]
    return recipe


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
                template=PromptTemplate(
                    template=LINK_TEMPLATES[key], input_variables=["state", "recipe"]
                ),
                llm=llm,
                parser=LINK_PARSERS[key],
                output_key=f"{key}_output",
            )
            links.append(link)

    return links
