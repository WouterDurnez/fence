"""
Templates classes for LLMs

This module contains classes for handling templates using Jinja2 for prompts.
"""

import re

from jinja2 import Template


class PromptTemplate:
    """
    A class for handling templates using Jinja2 for prompts.
    """

    def __init__(self, template: str, input_variables: list, **kwargs):
        """
        Initialize the PromptTemplate object.

        :param str template: The Jinja2 template string.
        :param list input_variables: List of input variables expected by the template.
        :param kwargs: Additional keyword arguments.
            - separator (str, optional): Separator to use when merging templates. Default is a space.
        """
        self.source = template
        self.input_variables = input_variables
        self.separator = kwargs.get("separator", " ")

    def __call__(self, **kwargs):
        """
        Alias for the render method to allow calling an instance as a function.

        :param kwargs: Keyword arguments to be passed to the render method.
        :return: Rendered template string.
        :rtype: str
        """
        return self.render(**kwargs)

    def __add__(self, *others):
        """
        Merge multiple PromptTemplate instances using the specified separator.

        :param others: Other PromptTemplate instances to be merged.
        :return: Merged PromptTemplate instance.
        :rtype: PromptTemplate

        :raises TypeError: If attempting to add a non-PromptTemplate object.
        """
        for other in others:
            if not isinstance(other, PromptTemplate):
                raise TypeError("Can only add PromptTemplates to PromptTemplate.")
            else:
                merged_template = self.source + self.separator + other.source
                merged_input_variables = list(
                    set(self.input_variables + other.input_variables)
                )
                return PromptTemplate(
                    template=merged_template, input_variables=merged_input_variables
                )
        else:
            raise TypeError("Can only add PromptTemplates to a PromptTemplate.")

    def _validate_template(self, template):
        """Check if input variables are in the template."""
        for variable in self.input_variables:
            if (
                "{{" + variable + "}}" not in template
                or "{{ " + variable + " }}" not in template
            ):
                raise ValueError(
                    f"Variable {variable} not found in template: {template}"
                )

    def render(self, input_dict: dict = None, **kwargs):
        """
        Render the template with the provided variables.

        :param kwargs: Keyword arguments containing values for the template variables.
        :return: Rendered template string.
        :rtype: str

        :raises ValueError: If any required variable is missing.
        """
        if input_dict:
            kwargs.update(input_dict)

        missing_variables = [
            variable for variable in self.input_variables if variable not in kwargs
        ]
        if missing_variables:
            raise ValueError(
                "Missing variables: {}".format(", ".join(missing_variables))
            )
        return Template(self.source).render(**kwargs)

    def __str__(self):
        return f"PromptTemplate(source={self.source}, input_variables={self.input_variables})"

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (
                self.source == other.source
                and self.input_variables == other.input_variables
            )
        return False

    def copy(self):
        return PromptTemplate(
            self.source, self.input_variables.copy(), separator=self.separator
        )

    @classmethod
    def from_file(cls, file_path):
        with open(file_path, "r") as file:
            template = file.read()
        pattern = r"{{\s*(\w+)\s*}}"
        input_variables = re.findall(pattern, template)
        return cls(template=template, input_variables=input_variables)


if __name__ == "__main__":
    test_template = PromptTemplate("test {{input}}", ["input"])
    test_template2 = PromptTemplate("lol {{input}}", ["input"])
    test_template3 = PromptTemplate("more lols {{input}}", ["input"])

    read_template = PromptTemplate.from_file("/data/some_prompt_template.txt")

    combined_template = test_template + test_template2 + test_template3

    a = combined_template.render(input="some input yo")

    looping_template_string = """{% for item in items %}{{item}}{% endfor %}"""
    looping_template = PromptTemplate(looping_template_string, ["items"])

    looped = looping_template.render(items=["a", "b", "c"])
