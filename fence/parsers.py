"""
PARSERS
transform LLM output (a string) into a more useful format (e.g. int, bool, dict)
"""

import json
import logging
import re
import tomllib
from abc import ABC, abstractmethod

from pydantic import ValidationError

logger = logging.getLogger(__name__)


################
# Base classes #
################


class BaseParser(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def parse(self, input_string: str):
        raise NotImplementedError


#####################
# BaseParser subclasses #
#####################


class IntParser(BaseParser):
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


class BoolParser(BaseParser):
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


class TripleBacktickParser(BaseParser):
    """
    A class to parse a string containing triple backticks.

    Example:
    ```
    This is a string contained in triple backticks.
    ```

    returns:

    This is a string contained in triple backticks.

    """

    def __init__(self, prefill: str | None = None):
        """
        Initialize the TripleBacktickParser with the given parameters.
        :param prefill: string to prefill the string with (as this was also passed as an assistant prefill in the prompt)
        """
        super().__init__()
        self.prefill = prefill

    def parse(self, input_string: str):
        """
        Parse a string containing triple backticks and return a dictionary.
        :param input_string: text string containing triple backticks
        :return: dictionary containing the data
        """

        # First, reapply the prefill string to the input string
        input_string = self.prefill + input_string if self.prefill else input_string

        # Extract the string from within the triple backticks
        pattern = re.compile(r"```([\s\S]*?)```")
        match = pattern.search(input_string)
        if not match:
            logger.warning(
                f"TripleBacktickParser expected triple backticks. Received: {input_string}."
            )
            raise ValueError("TripleBacktickParser found no triple backticks.")
        string = match.group(1).strip()

        return string


class TOMLParser(BaseParser):
    """
    A class to parse a string containing TOML data (possibly within triple backticks). Returns a dictionary with the key-value pairs.

    Example:
    ```
    some_key = "value"
    key1 = "value1"
    key2 = "value2"
    ```

    returns:

    {
        "some_key": "value",
        "key1": "value1",
        "key2": "value2"
    }
    """

    def __init__(self, triple_backticks: bool = True, prefill: str | None = None):
        """
        Initialize the TOML parser with the given parameters.
        :param triple_backticks: boolean indicating whether to extract the TOML string
        :param prefill: string to prefill the TOML string with (as this was also passed as an assistant prefill in the prompt)
        from within the triple backticks
        """
        super().__init__()
        self.triple_backticks = triple_backticks
        self.prefill = prefill

    def parse(self, input_string: str):
        """
        Parse a TOML string and return a dictionary.
        :param input_string: text string containing TOML
        :return: dictionary containing the TOML data
        """

        # First, reapply the prefill string to the input string, unless the input string already starts with the prefill
        input_string = (
            self.prefill + input_string
            if self.prefill and not input_string.startswith(self.prefill)
            else input_string
        )

        # If requested, extract the TOML string from within the triple backticks
        toml_string = (
            TripleBacktickParser().parse(input_string)
            if self.triple_backticks
            else input_string
        )

        # Strip the TOML string of the "toml" prefix
        if toml_string.startswith("toml"):
            toml_string = toml_string[4:]

        # Define a regular expression to match non-printable characters
        non_printable_regex = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\r]")

        # Use the regular expression to replace non-printable characters with an space
        toml_string = non_printable_regex.sub(" ", toml_string)

        # Load the TOML string into a dictionary
        toml_dict = tomllib.loads(toml_string)

        # Strip all string values of leading and trailing whitespace
        for key, value in toml_dict.items():
            if isinstance(value, str):
                toml_dict[key] = value.strip()

        return toml_dict


class PydanticParser(BaseParser):
    """
    A class to parse a string containing JSON data into a pydantic model.

    Example:
    ```json
    {
        "name": "John Doe",
        "age": 30,
        "email": "john@example.com"
    }
    ```

    With a corresponding pydantic model, returns an instance of that model.
    """

    def __init__(
        self, model_class, triple_backticks: bool = True, prefill: str | None = None
    ):
        """
        Initialize the PydanticParser with the given parameters.

        :param model_class: Pydantic model class to parse the data into
        :param triple_backticks: Whether to extract JSON from triple backticks
        :param prefill: String to prefill the JSON with (assistant prefill)
        """
        super().__init__()
        self.model_class = model_class
        self.triple_backticks = triple_backticks
        self.prefill = prefill

    def parse(self, input_string: str):
        """
        Parse a JSON string into a pydantic model instance.

        :param input_string: Text string containing JSON data
        :return: Instance of the specified pydantic model
        """
        # First, reapply the prefill string to the input string, unless it already starts with it
        input_string = (
            self.prefill + input_string
            if self.prefill and not input_string.startswith(self.prefill)
            else input_string
        )

        # If requested, extract the JSON string from within the triple backticks
        if self.triple_backticks:
            json_string = self._extract_from_backticks(input_string)
        else:
            json_string = input_string

        # Clean the JSON string
        json_string = self._clean_json_string(json_string)

        try:
            # Parse the JSON string into a dictionary
            json_dict = json.loads(json_string)

            # Create and validate the pydantic model instance
            model_instance = self.model_class(**json_dict)

            return model_instance

        except json.JSONDecodeError as e:
            logger.error(
                f"PydanticParser failed to parse JSON: {e}. Input: {json_string}"
            )
            raise ValueError(f"PydanticParser failed to parse JSON: {e}")

        except ValidationError as e:
            logger.error(f"PydanticParser validation failed: {e}. Input: {json_dict}")
            raise ValueError(f"PydanticParser validation failed: {e}")

        except Exception as e:
            logger.error(f"PydanticParser unexpected error: {e}. Input: {input_string}")
            raise ValueError(f"PydanticParser unexpected error: {e}")

    def _extract_from_backticks(self, input_string: str) -> str:
        """Extract content from triple backticks."""
        pattern = re.compile(r"```(?:json)?([\s\S]*?)```")
        match = pattern.search(input_string)
        if not match:
            logger.warning(
                f"PydanticParser expected triple backticks. Received: {input_string}"
            )
            raise ValueError("PydanticParser found no triple backticks.")
        return match.group(1).strip()

    def _clean_json_string(self, json_string: str) -> str:
        """Clean the JSON string of common issues."""
        # Strip the JSON string of language prefix
        if json_string.startswith("json"):
            json_string = json_string[4:].strip()

        # Define a regular expression to match non-printable characters (except newlines and tabs)
        non_printable_regex = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\r]")

        # Replace non-printable characters with a space
        json_string = non_printable_regex.sub(" ", json_string)

        return json_string.strip()
