"""
LINKS
are atomic LLM interactions. They transform input data into a prompt, send it to an LLM model, parse the output, and return the result.
"""

import logging
from abc import ABC, abstractmethod
from typing import Callable, Iterable

from fence.models.base import LLM
from fence.parsers import BaseParser
from fence.templates import MessagesTemplate, StringTemplate
from fence.utils.base import time_it

logger = logging.getLogger(__name__)

PROTECTED_LINK_KWARGS = ["model"]

################
# Base classes #
################


class BaseLink(ABC):
    """
    BaseLink class is a simple input-output building block. It is callable, and will call the run method. The run method
    takes a dictionary of input variables for the template, and returns a dictionary with the output under the specified
    key.
    """

    def __init__(self, output_key: str = "state", name: str = None):
        """
        Initialize the BaseLink object.

        :param input_keys: A list of input keys.
        :param output_key: The output key.
        """
        self.model = None
        self.input_keys = None
        self.output_key = output_key
        self.name = name

    def __call__(self, **kwargs):
        """
        Alias for the run method to allow calling an instance as a function.
        :param kwargs: Keyword arguments to be passed to the run method.
        :return: Output of the run method.
        """
        return self.run(**kwargs)

    @abstractmethod
    def run(self, input_dict: dict = None, **kwargs):
        """
        Run the link.
        :param input_dict: Values for the input variables.
        :param kwargs: Additional keyword arguments for the LLM model.
        :return:
        """
        raise NotImplementedError(
            "Base class <BaseLink> `run` method must be implemented."
        )

    def _validate_input(self, input_dict: dict):
        """
        Validate the input dictionary. Input dictionary should contain all the input keys for the Link.
        :param input_dict: Input dictionary.
        :return: True if the input dictionary is valid, False otherwise.
        """
        # Find both missing and superfluous variables
        missing_variables = [
            variable for variable in self.input_keys if variable not in input_dict
        ]
        superfluous_variables = [
            variable for variable in input_dict if variable not in self.input_keys
        ]
        if missing_variables:
            logger.warning(f"Missing variables: {missing_variables}")
        if superfluous_variables:
            logger.info(f"Superfluous variables: {superfluous_variables}")

    def __str__(self):
        return f"Link: {f'{self.name} ' if self.name else ''}<{self.input_keys}> -> <{self.output_key}>"

    def __repr__(self):
        return str(self)


###################
# Implementations #
###################


class TransformationLink(BaseLink):
    def __init__(
        self,
        input_keys: Iterable[str],
        output_key: str = "state",
        name: str = None,
        function: Callable = None,
    ):
        """
        Initialize the TransformationLink object.

        :param input_keys: A list of input keys.
        :param output_key: The output key.
        :param function: The transformation function.
        """
        super().__init__(output_key=output_key, name=name)
        self.input_keys = input_keys
        self.function = function

    @time_it
    def run(self, input_dict: dict = None, **kwargs) -> dict:
        """
        Run the link.
        :param input_dict: Values for the input variables.
        :param kwargs: Additional keyword arguments for the LLM model.
        :return:
        """
        logger.info(
            f"Executing {f'<{self.name}> ' if self.name else ''}Transformation Link"
        )

        # Validate the input dictionary
        self._validate_input(input_dict)

        # Call the transformation function
        response = self.function(input_dict)
        logger.debug(f"Response: {response}")

        # Build the response dictionary #
        #################################

        # We always return the output under the key 'output', but if the output key is different, we also return it
        # under that key. This allows us to track the output of a specific Link in a Chain, while still having a
        # consistent key for the output of the Link that allows subsequent Links to find the latest output.
        response_dict = {"state": response}
        if not self.output_key == "state":
            response_dict[self.output_key] = response
        return response_dict


class Link(BaseLink):
    """
    Link class that takes an LLM model, and calls it using the prompt template. Base blocks for building more complex
    chains. The Link class is callable, and will call the run method. The run method takes a dictionary of input
    variables for the template, and returns a dictionary with the output of the LLM model under the specified key.
    """

    def __init__(
        self,
        template: StringTemplate | MessagesTemplate,
        output_key: str = "state",
        model: LLM = None,
        name: str = None,
        parser: BaseParser = None,
    ):
        """
        Initialize the Link object.

        :param name: Name of the Link.
        :param model: An LLM model object.
        :param template: A PromptTemplate object.
        """
        self.template = template
        super().__init__(output_key=output_key, name=name)
        self.model = model
        self.template = template

        # Get the input keys from the template
        self.input_keys = template.input_variables
        self.parser = parser

    def __call__(self, **kwargs):
        """
        Alias for the run method to allow calling an instance as a function.
        :param kwargs: Keyword arguments to be passed to the run method.
        :return: Output of the run method.
        """
        return self.run(**kwargs)

    @time_it(threshold=30, only_warn=True)
    def run(self, input_dict: dict = None, **kwargs):
        """
        Run the link.
        :param input_dict: Variables for the template.
        :param kwargs: Additional keyword arguments for the LLM model.
        :return:
        """

        # Update the input dictionary with the keyword arguments
        if input_dict is None:
            input_dict = {}

        input_dict.update(kwargs)

        # Don't forget to call Mum
        logger.info(
            f"Executing {f'<{self.name}>' if self.name else 'unnamed'} Link "
            + (f"with input: [{input_dict}]" if input_dict else "without input")
        )

        # Check if an LLM model was provided
        if self.model is None and kwargs.get("model") is None:
            raise ValueError(
                "An model must be provided, either as an argument or in the Link object."
            )

        # Validate the input dictionary
        self._validate_input(input_dict)

        # Render the template
        prompt = self.template.render(input_dict=input_dict)
        logger.debug(f"Prompt: {prompt}")

        # Determine if the LLM model is provided as a keyword argument,
        # otherwise use the LLM model of the Link
        model = input_dict.pop("model", self.model)

        # Call the LLM model
        response = model.invoke(prompt=prompt)
        logger.debug(f"Raw response: {response}")

        # Parse the response
        if self.parser is not None:
            response = self.parser.parse(response)
            logger.debug(f"Parsed response: {response}")

        # Build the response dictionary #
        #################################

        # We always return the output under the key 'output', but if the output key is different, we also return it
        # under that key. This allows us to track the output of a specific Link in a Chain, while still having a
        # consistent key for the output of the Link that allows subsequent Links to find the latest output.

        # If the response is a dictionary, we assume it contains the primary output under the key 'state'
        if isinstance(response, dict):
            if "state" not in response.keys():
                logger.debug(
                    "Response is a dictionary, but does not contain the key "
                    "'state'. Using the full response as the state."
                )
                response_dict = {"state": response}
            else:
                response_dict = {"state": response["state"]}
        elif isinstance(response, (str, bool)):
            response_dict = {"state": response}
        else:
            raise TypeError(
                f"Response must be a dictionary, string or bool. Got {type(response)}."
            )

        # If the output key is not 'state', we also return the *full* output under that key
        if not self.output_key == "state":
            response_dict[self.output_key] = response

        logger.debug(f"Current state: {response_dict['state']}")

        return response_dict
