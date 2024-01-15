import re
import tomllib

VALID_FLAVORS = ["sarcastic", "formal", "informal"]
VALID_VERBOSITY = ["shorter", "longer"]


def parse_toml(toml_string: str):
    """
    Parse a TOML string and return a dictionary.
    :param toml_string: text string containing TOML
    :return: dictionary containing the TOML data
    """

    # Extract the TOML string from within the triple backticks
    pattern = re.compile(r"```([\s\S]*?)```")
    match = pattern.search(toml_string)
    toml_string = match.group(1)

    # Strip the TOML string of the "toml" prefix
    if toml_string.startswith("toml"):
        toml_string = toml_string[4:]

    # Load the TOML string into a dictionary
    toml_dict = tomllib.loads(toml_string)

    # Strip all string values of leading and trailing whitespace
    for key, value in toml_dict.items():
        if isinstance(value, str):
            toml_dict[key] = value.strip()

    return toml_dict


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
        if not isinstance(recipe["policy"], str):
            raise ValueError(f"Invalid policy: {recipe['policy']}")
    return True
