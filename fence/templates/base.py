import logging
import re
from abc import ABC, abstractmethod

from fence.templates.models import Messages

logger = logging.getLogger(__name__)


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
            raise TypeError("Template must be a string or a Messages object.")

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
        Find placeholders in a list of messages. Placeholders are defined as text enclosed in double curly braces.
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
            raise ValueError(f"Missing variables: {missing_variables}")
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

        return re.findall(r"\{\s*([a-zA-Z0-9_]+)\s*}", text)

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

        return text.format(**input_dict)

    def __str__(self):
        return f"{self.__class__}: {self.source}"
