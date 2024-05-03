######################
# LLM output parsers #
######################


import re
import tomllib
from abc import ABC, abstractmethod

################
# Base classes #
################


class Parser(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def parse(self, input_string: str):
        raise NotImplementedError


#####################
# Parser subclasses #
#####################


class BoolParser(Parser):
    """
    A class to parse a string containing a boolean value. Returns a boolean.

    Example:
    "true"

    returns:
    True
    """

    def parse(self, input_string: str):
        """
        Parse a string containing a boolean value and return a boolean.

        :param input_string: text string containing a boolean value
        :return: boolean value
        """
        input_string = input_string.strip().lower()
        if input_string in ["true", "t", "yes", "y", "1"]:
            return True
        elif input_string in ["false", "f", "no", "n", "0"]:
            return False
        else:
            raise ValueError(f"Invalid boolean value: {input_string}")


class TripleBacktickParser(Parser):
    def parse(self, input_string: str):
        """
        Parse a string containing triple backticks and return a dictionary.
        :param input_string: text string containing triple backticks
        :return: dictionary containing the data
        """
        # Extract the string from within the triple backticks
        pattern = re.compile(r"```([\s\S]*?)```")
        match = pattern.search(input_string)
        string = match.group(1).strip()

        return string


class TOMLParser(Parser):
    def parse(self, input_string: str, triple_backticks: bool = True):
        """
        Parse a TOML string and return a dictionary.
        :param input_string: text string containing TOML
        :return: dictionary containing the TOML data
        """

        # If requested, extract the TOML string from within the triple backticks
        toml_string = (
            TripleBacktickParser().parse(input_string)
            if triple_backticks
            else input_string
        )

        # Strip the TOML string of the "toml" prefix
        if toml_string.startswith("toml"):
            toml_string = toml_string[4:]

        # Define a regular expression to match non-printable characters
        non_printable_regex = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\r]")

        # Use the regular expression to replace non-printable characters with an empty string
        toml_string = non_printable_regex.sub("", toml_string)

        # Load the TOML string into a dictionary
        try:
            toml_dict = tomllib.loads(toml_string)
        except Exception as e:
            raise ValueError(f"Error parsing TOML: {e}")

        # Strip all string values of leading and trailing whitespace
        for key, value in toml_dict.items():
            if isinstance(value, str):
                toml_dict[key] = value.strip()

        return toml_dict
