"""
PARSERS
transform LLM output (a string) into a more useful format (e.g. int, bool, dict)
"""

import re
import tomllib
from abc import ABC, abstractmethod

from fence.utils.base import setup_logging

logger = setup_logging(__name__)

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


class IntParser(Parser):
    """
    A class to parse a string containing an integer. Returns an integer.

    Example:
    "123"

    returns:
    123
    """

    def parse(self, input_string: str):
        """
        Parse a string containing an integer and return an integer.

        :param input_string: text string containing an integer
        :return: integer value
        """

        # Extract the integer value from the input string
        int_vals = re.findall(r"[-+]?\d+", input_string)

        # Check if an integer value was found
        if len(int_vals) == 0:
            logger.error(
                f"IntParser expected an integer value. Received: {input_string}."
            )
            raise ValueError("IntParser found no integer value.")

        # Check if multiple integer values were found
        if len(int_vals) > 1:
            logger.error(
                f"IntParser expected a single integer value. Received: {input_string}."
            )
            raise ValueError("IntParser found multiple integer values.")

        return int(int_vals[0])


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

        pattern = r"\b(true|false)\b"

        # Extract the boolean value from the input string
        bool_vals = {
            val.lower()
            for val in re.findall(pattern, input_string, flags=re.IGNORECASE)
        }

        # Check if positive value is present
        if "true" in bool_vals:

            # Check for ambiguous response
            if "false" in bool_vals:
                logger.error(
                    f"Ambiguous response. Both 'true' and 'false' in received: {input_string}."
                )
                raise ValueError("BoolParser found ambiguous response.")

            # Otherwise, return True
            return True

        # Check if negative value is present
        elif "false" in bool_vals:
            return False

        raise ValueError(
            f"BooleanOutputParser expected output value to include either "
            f"'true' or 'false'. Received: {input_string}"
        )


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
    def parse(
        self, input_string: str, pre_fill="```toml\n", triple_backticks: bool = True
    ):
        """
        Parse a TOML string and return a dictionary.
        :param input_string: text string containing TOML
        :return: dictionary containing the TOML data
        """

        # If pre_fill is provided, add it in front of the input string
        if pre_fill:
            input_string = pre_fill + input_string

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


if __name__ == "__main__":

    test_string = "test 12. test"
    int_parser = IntParser()
    print(int_parser.parse(test_string))
