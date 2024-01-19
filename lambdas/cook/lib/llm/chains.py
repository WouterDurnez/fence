import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Iterable, Callable

from lambdas.cook.lib.llm.models import LLM, ClaudeInstantLLM
from lambdas.cook.lib.llm.parsers import Parser
from lambdas.cook.lib.llm.templates import PromptTemplate
from lambdas.cook.lib.utils.base import setup_logging

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

    def __init__(self, input_keys: Iterable[str], output_key: str = "state", name: str = None):
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
    def run(self, input_dict: dict, **kwargs):
        """
        Run the link.
        :param input_dict: Values for the input variables.
        :param kwargs: Additional keyword arguments for the LLM model.
        :return:
        """
        raise NotImplementedError("Base class <BaseLink> `run` method must be implemented.")

class BaseChain(ABC):
    """
    BaseChain class that takes a list or set of Link objects, determines the order in which to run them, and does so.
    """

    def __init__(self, links: Iterable[BaseLink], llm: LLM=None):
        """
        Initialize the BaseChain object.

        :param links: A set of Link objects.
        """
        self.links = links
        self.llm = llm

    def __call__(self, *args, **kwargs):
        """
        Alias for the run method to allow calling an instance as a function.
        :param kwargs: Keyword arguments to be passed to the run method.
        :return: Output of the run method.
        """
        return self.run(*args, **kwargs)

    def _find_and_validate_keys(self, input_keys):
        """
        Find all input and output keys in the chain set, determine which input keys are required, and validate the input keys.
        Required keys do not appear as output keys in any other chain.
        """
        input_keys_set = set(
            input_variable
            for link in self.links
            for input_variable in link.input_keys
        )
        output_keys = set(link.output_key for link in self.links)
        required_keys = input_keys_set - output_keys

        # Validate input keys
        if input_keys is None:
            raise ValueError(f"The following input keys are required: {required_keys}")
        elif not isinstance(input_keys, Iterable):
            raise TypeError("Input_keys must be an iterable")

        # Check if all required keys are present
        if not required_keys.issubset(input_keys):
            raise ValueError(f"The following input keys are required: {required_keys}")

    @abstractmethod
    def run(self, input_dict):
        """
        Run the Links in the correct order based on the dependencies in the graph.

        :param input_dict: Variables for the template.
        :return: Dictionary containing all the input and output variables.
        """
        raise NotImplementedError("Base class <BaseChain> `run` method must be implemented.")

################
# Link classes #
################

class TransformationLink(BaseLink):

    def __init__(self, input_keys: Iterable[str], output_key: str = "output", function: Callable = None):
        """
        Initialize the TransformationLink object.

        :param input_keys: A list of input keys.
        :param output_key: The output key.
        :param function: The transformation function.
        """
        super().__init__(input_keys, output_key)
        self.function = function

    def run(self, input_dict: dict, **kwargs):
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
        response_dict = {'output': response}
        if not self.output_key == 'output':
            response_dict[self.output_key] = response
        return response_dict

class Link(BaseLink):
    """
    Link class that takes an LLM model, and calls it using the prompt template. Base blocks for building more complex
    chains. The Link class is callable, and will call the run method. The run method takes a dictionary of input
    variables for the template, and returns a dictionary with the output of the LLM model under the specified key.
    """

    def __init__(self, template: PromptTemplate, output_key: str = "state", llm: LLM = None,
                 name: str = None, parser: Parser = None):
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

    def run(self, input_dict: dict, **kwargs):
        """
        Run the link.
        :param input_dict: Variables for the template.
        :param kwargs: Additional keyword arguments for the LLM model.
        :return:
        """

        logger.info(f"Running Link: {self.name}")
        logger.info(f"🔑Input keys: {self.input_keys}")

        # Check if an LLM model was provided
        if self.llm is None and kwargs.get("llm") is None:
            raise ValueError("An LLM model must be provided.")

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
            logger.info(f"Parsed response: {response}")

        # Build the response dictionary #
        #################################

        # We always return the output under the key 'output', but if the output key is different, we also return it
        # under that key. This allows us to track the output of a specific Link in a Chain, while still having a
        # consistent key for the output of the Link that allows subsequent Links to find the latest output.

        # If the response is a dictionary, we assume it contains the primary output under the key 'output'
        if isinstance(response, dict):
            if not 'output' in response.keys():
                logger.warning(f"Response is a dictionary, but does not contain the key 'state'. Using the full response as the state.")
                response_dict = {'state': response}
            else:
                response_dict = {'state': response['output']}
        elif isinstance(response, str):
            response_dict = {'state': response}
        else:
            raise TypeError(f"Response must be a dictionary or a string. Got {type(response)}.")

        # If the output key is not 'output', we also return the *full* output under that key
        if not self.output_key == 'state':
            response_dict[self.output_key] = response

        logger.info(f"🔑 Output key: {list(response_dict.keys())}")
        return response_dict

    def __str__(self):
        return f"Link: {f'{self.name} ' if self.name else ''}<{self.template.input_variables}> -> <{self.output_key}>"

    def __repr__(self):
        return str(self)


#################
# Chain classes #
#################

class Chain(BaseChain):
    """
    Chain class that takes a list or set of Link objects, determines the order in which to run them, and does so.
    """

    def __init__(self, links: Iterable[Link], llm: LLM=None):
        """
        Initialize the Chain object.

        :param links: A set of Link objects.
        """
        super().__init__(links=links, llm=llm)
        self.graph = defaultdict(list)

        # Build the graph
        self._build_graph()

    def _build_graph(self):
        """
        Get the input and output variables of each Link in the Chain, and build a graph of dependencies.
        """
        for link in self.links:
            current_output_key = link.output_key
            dependent_links = [
                other_link
                for other_link in self.links
                if current_output_key in other_link.input_keys
            ]
            self.graph[link].extend(dependent_links)

    def run(self, input_dict):
        """
        Run the Links in the correct order based on the dependencies in the graph.

        :param input_dict: Variables for the template.
        :return: Dictionary containing all the input and output variables.
        """

        # Check if an LLM model was provided
        if self.llm is None and any(link.llm is None for link in self.links):
            raise ValueError("An LLM model must be provided.")

        # Validate the input keys
        self._find_and_validate_keys(input_dict)

        # Initialize the output dictionary with the input dictionary, which will be passed to the first chain
        output_dict = input_dict

        # Iterate through Links in topological order
        for link in self._topological_sort():
            # Run the Link with Chain's LLM model if the Link doesn't have one
            if link.llm is None:
                output_dict.update(link.run(input_dict=output_dict, llm=self.llm))
            else:
                output_dict.update(link.run(input_dict=output_dict))

        return output_dict

    def _topological_sort(self):
        """
        Perform topological sorting of the Links based on the dependencies in the graph.
        """
        visited = set()
        stack = set()
        sorted_links = []

        def depth_first_search(link: BaseLink):
            """
            Perform depth-first search on the graph.
            :param link: Link to start the search from.
            :return: None
            """
            nonlocal visited, stack, sorted_links

            # If the link is already in the stack, we are revisiting it,
            # which means there is a cycle
            if link in stack:
                raise ValueError("Cycle detected in the dependency graph.")

            # Move on to next unvisited link
            if link not in visited:
                visited.add(link)
                stack.add(link)

                # Recursively visit all dependent links
                for dependent_link in self.graph[link]:
                    depth_first_search(dependent_link)

                stack.remove(link)
                sorted_links.insert(0, link)

        for link in self.links:
            depth_first_search(link)

        return sorted_links


    def __call__(self, *args, **kwargs):
        """
        Alias for the run method to allow calling an instance as a function.
        :param kwargs: Keyword arguments to be passed to the run method.
        :return: Output of the run method.
        """
        return self.run(*args, **kwargs)


class LinearChain(BaseChain):
    """
    LinearChain class that takes a list or set of Link objects, and runs them in the order they are provided.
    """

    def __init__(self, links: Iterable[BaseLink], llm=None):
        """
        Initialize the LinearChain object.

        :param links: A set of Link objects.
        """
        super().__init__(links=links, llm=llm)

    def run(self, input_dict):

        # Check if an LLM model was provided
        if self.llm is None and any(link.llm is None for link in self.links):
            raise ValueError("An LLM model must be provided.")

        # Validate the input keys
        self._find_and_validate_keys(input_dict)

        # Initialize the output dictionary with the input dictionary, which will be passed to the first chain
        state_dict = input_dict

        # Iterate through Links in order
        for link in self.links:

            logger.info(f"⚙️State keys going in link {link.name}: {list(state_dict.keys())}")

            # Check if state_dict contains all the input keys for the Link
            if not all(input_key in state_dict for input_key in link.input_keys):
                raise ValueError(f"Input keys {link.input_keys} not found in state_dict.")

            # Run the Link with Chain's LLM model if the Link doesn't have one
            if link.llm is None:
                response_dict = link.run(input_dict=state_dict, llm=self.llm)
            else:
                response_dict = link.run(input_dict=state_dict)

            # Update the state_dict with the output of the Link
            state_dict.update(response_dict)

            logger.info(f"⚙️State keys going out of link {link.name}: {list(state_dict.keys())}")

        return state_dict

if __name__ == "__main__":
    # Instantiate an LLM model
    claude = ClaudeInstantLLM(source="test")

    # Example chains
    link_a = Link(
        template=PromptTemplate(
            "What's the opposite of {{A}}? Reply with one word.", ["A"]
        ),
        output_key="X",
    )
    link_b = Link(
        template=PromptTemplate(
            "Write a very short story about {{B}}. Two sentences max.", ["B"]
        ),
        output_key="Y",
    )
    link_c = Link(
        template=PromptTemplate(
            "Write a sarcastic poem using this as inspiration: '{{X}}' and '{{Y}}'",
            ["X", "Y"],
        ),
        output_key="output",
    )

    # Combine the chains into a set
    chain = Chain(llm=claude, links=[link_a, link_b, link_c])

    # Run the chain set
    result = chain(input_dict={"A": "calm", "B": "a hurricane"})

    print(result["output"])

    # Example ConcatLink
    concatenate = lambda x,y: f"{x} {y}"
    concat_link = TransformationLink(input_keys=["A", "B"], function=concatenate, output_key="C")
    concat_link2 = TransformationLink(input_keys=["C", "D"], function=concatenate)
    concat_link3 = TransformationLink(input_keys=["X", "Y"], function=concatenate)

    # Example LinearChain
    linear_chain = LinearChain(llm=claude, links=[concat_link, concat_link2])
    result = linear_chain(input_dict={"A": "I am", "B": "a calm", "D": "hurricane"})

    # Let's try another one
    linear_chain = LinearChain(llm=claude, links=[link_a, link_b, concat_link3])
    result_linear = linear_chain(input_dict={"A": "I am", "B": "a calm", "D": "hurricane"})

    # Let's see what error a BaseLink run method raises
    base_link = BaseLink(input_keys=["A"], output_key="B")
    try:
        base_link.run(input_dict={"A": "test"})
    except Exception as e:
        print(e)
