import logging
import re
import string
from abc import ABC, abstractmethod

from fence.templates.models import Messages

logger = logging.getLogger(__name__)


class SafeFormatter(string.Formatter):
    """
    A custom string formatter that safely handles missing keys and nested attributes.

    This formatter extends the built-in string.Formatter to provide the following features:
    1. Gracefully handles missing keys by returning the original placeholder.
    2. Supports nested attribute and dictionary key access (e.g., {user.name} or {dict.key}).
    3. Returns original placeholders for any unresolved fields.

    Usage:
        formatter = SafeFormatter()
        result = formatter.format("Hello, {name}! Your score is {user.score}{test.ignore}.", name="Alice", user={"score": 95})

    """

    def get_field(self, field_name, args, kwargs):
        """
        Retrieve the value for a given field name.

        This method handles nested attribute access and gracefully falls back to the original
        placeholder if the field is not found.

        :param field_name: The name of the field to retrieve.
        :return: A pair (obj, used_key), where obj is the retrieved object or the original
        """
        try:
            # Start with the kwargs dictionary
            obj = kwargs

            # Split the field_name into parts for nested access
            for name in field_name.split("."):
                if isinstance(obj, dict):
                    # If obj is a dictionary, use get() to safely retrieve the next item
                    obj = obj.get(name, f"{{{field_name}}}")
                else:
                    # If obj is not a dictionary, try to get the attribute
                    # If the attribute doesn't exist, return the original placeholder
                    obj = getattr(obj, name, f"{{{field_name}}}")

            return obj, field_name
        except Exception:
            # If any exception occurs during the process, return the original placeholder
            return f"{{{field_name}}}", field_name

    def format_field(self, value, format_spec):
        """
        Format a field based on the given format_spec.

        This method overrides the default behavior to return original placeholders as-is.

        :param value: The value to format.
        :param format_spec: The format specification.
        :return: The formatted value or the original placeholder if it was not resolved.
        """
        if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
            # If the value looks like an unresolved placeholder, return it as-is
            return value
        # Otherwise, use the default formatting behavior
        return super().format_field(value, format_spec)


class BaseTemplate(ABC):
    """
    Base class for template objects.
    """

    def __init__(self, source: str | Messages, **kwargs):
        """
        Initialize the BaseTemplate object.

        :param str | Messages source: The template string, or a list of Message objects.
        :param list input_variables: List of input variables expected by the template.
        :param kwargs: Additional keyword arguments.
            - separator (str, optional): Separator to use when merging string-based templates. Default is a space.
        """

        self.source = source
        self.type = type(source)
        self.input_variables = []

        # We have no time for anything other than strings or Messages
        if not isinstance(source, (str, Messages)):
            raise TypeError(
                f"Template must be a string or a Messages object - received {type(source)}"
            )

    def __call__(self, input_dict: dict = None, **kwargs):
        """
        Render the template with the provided variables.

        :param kwargs: Keyword arguments containing values for the template variables.
        :return: Rendered template string.
        :rtype: str

        :raises ValueError: If any required variable is missing.
        """
        return self.render(input_dict=input_dict, **kwargs)

    @abstractmethod
    def render(self, input_dict: dict = None, **kwargs):
        """
        Render the template using the input variables.

        :param input_dict: Dictionary of input variables.
        :param kwargs: Input variables for the template.
        :return: The rendered template.
        """
        pass

    @abstractmethod
    def _find_placeholders(self) -> list[str]:
        """
        Find placeholders in a list of messages. Placeholders are defined as text enclosed in single curly braces.
        They are used to denote variables that need to be replaced in the messages.
        """
        pass

    def _validate_input(self, input_dict: dict):

        # Find both missing and superfluous variables
        missing_variables = [
            variable for variable in self.input_variables if variable not in input_dict
        ]
        superfluous_variables = [
            variable for variable in input_dict if variable not in self.input_variables
        ]
        if missing_variables:
            logger.warning(f"Possible missing variables: {missing_variables}")
        if superfluous_variables:
            logger.debug(f"Superfluous variables: {superfluous_variables}")

    ###############
    # Class utils #
    ###############

    def _find_placeholders_in_text(self, text: str) -> list[str]:
        """
        Find placeholders in a text.

        :param str text: The text.
        :return: List of placeholders
        :rtype: list[str]
        """

        return re.findall(r"\{\s*([a-zA-Z0-9_.]+)\s*}", text)

    def _render_string(self, text: str, input_dict: dict = None, **kwargs):
        """
        Render a string template with the provided variables.

        :param str text: The template string.
        :param dict input_dict: Dictionary of input variables.
        :param kwargs: Keyword arguments containing values for the template variables.
        :return: Rendered string.
        :raises ValueError: If any required variable is missing.
        """

        # Make input_dict
        if input_dict is None:
            input_dict = {}

        input_dict.update(kwargs)

        # Find both missing and superfluous variables
        self._validate_input(input_dict=input_dict)

        # Format the string
        formatter = SafeFormatter()

        return formatter.format(text, **input_dict)

    def __str__(self):
        return f"{self.__class__}: {self.source}"
