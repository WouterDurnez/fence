"""
This module contains the StringTemplate class, i.e. a vanilla prompt template class.
"""

from fence.templates.base import BaseTemplate


class StringTemplate(BaseTemplate):

    def __init__(self, source: str, **kwargs):
        """
        Initialize the MessagesTemplate object.

        :param str | Messages source: The template string, or a list of Message objects.
        :param list input_variables: List of input variables expected by the template.
        :param kwargs: Additional keyword arguments.
            - separator (str, optional): Separator to use when merging string-based templates. Default is a space.
        """

        super().__init__(source=source)

        # Find placeholders in the template
        self.input_variables = self._find_placeholders()

        # Separator for string concatenation
        self.separator = kwargs.get("separator", " ")

    def render(self, input_dict: dict = None, **kwargs):
        """
        Render the template using the input variables. Alias for the render method.

        :param input_dict: Dictionary of input variables.
        :param kwargs: Input variables for the template.
        :return: The rendered template.
        """

        return self._render_string(text=self.source, input_dict=input_dict, **kwargs)

    def __call__(self, *args, **kwargs):
        """
        Render the template with the provided variables.

        :param kwargs: Keyword arguments containing values for the template variables.
        :return: Rendered template string.
        :rtype: str

        :raises ValueError: If any required variable is missing.
        """
        return self.render(*args, **kwargs)

    # Helper functions
    def _find_placeholders(self) -> list[str]:
        """
        Find placeholders in a list of messages. Placeholders are defined as text enclosed in double curly braces.
        They are used to denote variables that need to be replaced in the messages.

        :param Messages messages: The list of messages.
        :return: List of placeholders.
        :rtype: list[str]
        """

        placeholders = self._find_placeholders_in_text(text=self.source)

        # Remove duplicates
        placeholders = list(set(placeholders))

        return placeholders

    def __str__(self):
        return f"StringTemplate: {self.source}"

    def __repr__(self):
        return f"StringTemplate(template={self.source}, input_variables={self.input_variables})"

    def __add__(self, other):
        """
        Combine two PromptTemplate instances.

        :param PromptTemplate other: The other PromptTemplate instance.
        :return: The combined PromptTemplate instance.
        """

        if not isinstance(other, StringTemplate):
            raise ValueError("Can only combine PromptTemplate instances.")

        return StringTemplate(source=self.source + self.separator + other.source)

    def __eq__(self, other):
        """
        Check if two PromptTemplate instances are equal.

        :param PromptTemplate other: The other PromptTemplate instance.
        :return: True if the two instances are equal, False otherwise.
        """

        if not isinstance(other, StringTemplate):
            return False

        return (
            self.source == other.source
            and self.input_variables == other.input_variables
        )

    def copy(self):
        """
        Create a copy of the PromptTemplate instance.
        """
        return StringTemplate(source=self.source, separator=self.separator)
