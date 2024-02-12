import logging
from abc import ABC, abstractmethod
from typing import Iterable, Callable

from fence.src.llm.templates import PromptTemplate
from fence.src.llm.models import LLM
from fence.src.llm.parsers import Parser
from fence.src.utils.base import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

################
# Base classes #
################


class BaseLink(ABC):
    """
    BaseLink class is a simple input-output building block. It is callable, and will call the run method. The run method
    takes a dictionary of input variables for the template, and returns a dictionary with the output under the specified
    key.
    """

    def __init__(
        self, input_keys: Iterable[str], output_key: str = "state", name: str = None
    ):
        """
        Initialize the BaseLink object.

        :param input_keys: A list of input keys.
        :param output_key: The output key.
        """
        self.llm = None
        self.input_keys = input_keys if input_keys is not None else []
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


###################
# Implementations #
###################


class TransformationLink(BaseLink):
    def __init__(
        self,
        input_keys: Iterable[str],
        output_key: str = "state",
        function: Callable = None,
    ):
        """
        Initialize the TransformationLink object.

        :param input_keys: A list of input keys.
        :param output_key: The output key.
        :param function: The transformation function.
        """
        super().__init__(input_keys, output_key)
        self.function = function

    def run(self, input_dict: dict = None, **kwargs):
        """
        Run the link.
        :param input_dict: Values for the input variables.
        :param kwargs: Additional keyword arguments for the LLM model.
        :return:
        """

        # Get the input variables
        input_variables = [input_dict[input_key] for input_key in self.input_keys]

        # Call the transformation function
        response = self.function(*input_variables)
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
        template: PromptTemplate,
        output_key: str = "state",
        llm: LLM = None,
        name: str = None,
        parser: Parser = None,
    ):
        """
        Initialize the Link object.

        :param name: Name of the Link.
        :param llm: An LLM model object.
        :param template: A PromptTemplate object.
        """
        self.template = template
        super().__init__(template.input_variables, output_key)
        self.name = name
        self.llm = llm
        self.template = template
        self.parser = parser

    def __call__(self, **kwargs):
        """
        Alias for the run method to allow calling an instance as a function.
        :param kwargs: Keyword arguments to be passed to the run method.
        :return: Output of the run method.
        """
        return self.run(**kwargs)

    def run(self, input_dict: dict = None, **kwargs):
        """
        Run the link.
        :param input_dict: Variables for the template.
        :param kwargs: Additional keyword arguments for the LLM model.
        :return:
        """

        logger.debug(
            f"🖇️ Executing {(f'<{self.name}>') if self.name else ''} Link with input: {input_dict}"
        )

        # Check if an LLM model was provided
        if self.llm is None and kwargs.get("llm") is None:
            raise ValueError("An LLM model must be provided.")

        # Check if the input_dict contains all the input keys for the Link
        if not all(input_key in input_dict for input_key in self.input_keys):
            raise ValueError(
                f"Input keys {self.input_keys} not found in input_dict: {input_dict}"
            )

        # Render the template
        prompt = self.template(**input_dict)
        logger.debug(f"Prompt: {prompt}")

        # Determine if the LLM model is provided as a keyword argument,
        # otherwise use the LLM model of the Link
        llm = kwargs.pop("llm", self.llm)

        # Call the LLM model
        response = llm(prompt, **kwargs)
        logger.debug(f"Response: {response}")

        # Parse the response
        if self.parser is not None:
            response = self.parser.parse(response)
            logger.debug(f"Parsed response: {response}")

        # Build the response dictionary #
        #################################

        # We always return the output under the key 'output', but if the output key is different, we also return it
        # under that key. This allows us to track the output of a specific Link in a Chain, while still having a
        # consistent key for the output of the Link that allows subsequent Links to find the latest output.

        # If the response is a dictionary, we assume it contains the primary output under the key 'output'
        if isinstance(response, dict):
            if "state" not in response.keys():
                logger.debug(
                    "Response is a dictionary, but does not contain the key 'state'. Using the full response as the state."
                )
                response_dict = {"state": response}
            else:
                response_dict = {"state": response["state"]}
        elif isinstance(response, str):
            response_dict = {"state": response}
        else:
            raise TypeError(
                f"Response must be a dictionary or a string. Got {type(response)}."
            )

        # If the output key is not 'state', we also return the *full* output under that key
        if not self.output_key == "state":
            response_dict[self.output_key] = response

        logger.debug(f"🧐 Current state: {response_dict['state']}")

        return response_dict

    def __str__(self):
        return f"Link: {f'{self.name} ' if self.name else ''}<{self.template.input_variables}> -> <{self.output_key}>"

    def __repr__(self):
        return str(self)
