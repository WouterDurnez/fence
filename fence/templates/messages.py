"""
This module contains the MessagesTemplate class, which is used to render a list of messages with the provided variables.
The module was created to accommodate the Claude3 model API format.
"""

from fence.templates.base import BaseTemplate
from fence.templates.models import Message, Messages, TextContent
from fence.templates.string import StringTemplate


class MessagesTemplate(BaseTemplate):

    def __init__(self, source: str | Messages, **kwargs):
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

    def render(self, input_dict: dict = None, **kwargs):
        """
        Render a list of messages with the provided variables.

        :param Messages messages: List of Message objects.
        :param dict input_dict: Dictionary of input variables.
        :param kwargs: Keyword arguments containing values for the template variables.
        :return: Rendered list of messages.
        :rtype: Messages

        :raises ValueError: If any required variable is missing.
        """

        # Make input_dict
        if input_dict is None:
            input_dict = {}
        input_dict.update(kwargs)

        # Find both missing and superfluous variables
        self._validate_input(input_dict=input_dict)

        # Messages are the source
        messages = self.source

        # First get the system message, if any, and render it
        system = None
        if messages.system:
            system = self._render_string(messages.system, input_dict=input_dict)

        # Then go through each user/assistant message
        rendered_messages = []

        for message in messages.messages:

            # If content is a string, render it
            if isinstance(message.content, str):
                content = self._render_string(message.content, input_dict=input_dict)

            # If content is a list of content objects...
            elif isinstance(message.content, list):

                # ...go through each content object and render text content
                content = []
                for content_object in message.content:
                    if content_object.type == "text":
                        content.append(
                            TextContent(
                                text=self._render_string(
                                    content_object.text, input_dict=input_dict
                                )
                            )
                        )
                    elif content_object.type == "image":
                        content.append(content_object)

            # If content is neither a string nor a list, raise an error
            else:
                raise TypeError(
                    "Content must be a string or a list of content objects."
                )

            rendered_messages.append(Message(role=message.role, content=content))

        return Messages(messages=rendered_messages, system=system)

    # Helper functions
    def _find_placeholders(self) -> list[str]:
        """
        Find placeholders in a list of messages. Placeholders are defined as text enclosed in double curly braces.
        They are used to denote variables that need to be replaced in the messages.

        :param Messages messages: The list of messages.
        :return: List of placeholders.
        :rtype: list[str]
        """

        # Initialize placeholders
        placeholders = set()

        # ...otherwise, the template is a Messages object
        messages = self.source

        # Check the system message
        if messages.system:
            placeholders.update(self._find_placeholders_in_text(text=messages.system))

        # Then go through each user/assistant message
        for message in messages.messages:

            # Content can be a string, or a list of content objects. If it's a list...
            if isinstance(message.content, list):

                # ...go through each content object and find placeholders in text content
                for content in message.content:
                    if content.type == "text":
                        placeholders.update(
                            self._find_placeholders_in_text(text=content.text)
                        )

            # If content is a string, find placeholders in the string
            elif isinstance(message.content, str):
                placeholders.update(
                    self._find_placeholders_in_text(text=message.content)
                )

            # If content is neither a string nor a list, raise an error
            else:
                raise TypeError(
                    "Content must be a string or a list of content objects."
                )

        return list(placeholders)

    def to_string_template(self) -> StringTemplate:
        """
        Convert the MessagesTemplate object to a StringTemplate object.

        :return: StringTemplate object.
        :rtype: StringTemplate
        """

        # If the source is a string, return a StringTemplate object
        if isinstance(self.source, str):
            return StringTemplate(source=self.source)

        # If the source is a Messages object, convert it to a string
        messages = self.source

        # Initialize the string
        new_source = ""

        # Add the system message
        if messages.system:
            new_source += messages.system + "\n"

        # Add the user/assistant messages
        for message in messages.messages:
            if isinstance(message.content, str):
                new_source += message.content + "\n"
            elif isinstance(message.content, list):
                for content in message.content:
                    if content.type == "text":
                        new_source += content.text + "\n"

        return StringTemplate(source=new_source)

    def __str__(self):
        return f"MessagesTemplate: {self.source}"

    def __repr__(self):
        return f"MessagesTemplate(template={self.source}, input_variables={self.input_variables})"

    def __add__(self, other):
        """
        Combine two MessagesTemplate instances.
        :param other: other MessagesTemplate instance
        :return: Combined MessagesTemplate instance
        """

        if not isinstance(other, MessagesTemplate):
            raise TypeError("Can only combine MessagesTemplate instances.")

        # Combine the messages
        system = self.source.system + " " + other.source.system
        messages = self.source.messages + other.source.messages

        return MessagesTemplate(Messages(system=system, messages=messages))

    def __eq__(self, other):
        """
        Check if two MessagesTemplate instances are equal.
        :param other: other MessagesTemplate instance
        :return: True if the two instances are equal, False otherwise
        """

        if not isinstance(other, MessagesTemplate):
            return False

        return (
            self.source == other.source
            and self.input_variables == other.input_variables
        )
