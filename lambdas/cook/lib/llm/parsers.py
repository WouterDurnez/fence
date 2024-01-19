import re
import tomllib
from abc import ABC, abstractmethod


# Parser base class
class Parser(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def parse(self, input_string: str):
        pass


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
    def parse(self, input_string: str):
        """
        Parse a TOML string and return a dictionary.
        :param input_string: text string containing TOML
        :return: dictionary containing the TOML data
        """

        # Extract the TOML string from within the triple backticks
        pattern = re.compile(r"```([\s\S]*?)```")
        match = pattern.search(input_string)
        toml_string = match.group(1).strip()

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
