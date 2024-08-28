"""
CHAINS
combine and organize links into sequences of LLM operations
"""

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Collection, Iterable

from fence.links import BaseLink, Link
from fence.models.base import LLM
from fence.models.claude import ClaudeInstant
from fence.templates import StringTemplate
from fence.utils.base import time_it

logger = logging.getLogger(__name__)


################
# Base classes #
################


class BaseChain(ABC):
    """
    BaseChain class that takes a list or set of Link objects, determines the order in which to run them, and does so.
    """

    def __init__(
        self, links: Collection[BaseLink] | Iterable[BaseLink], llm: LLM = None
    ):
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

    @time_it
    def _find_and_validate_keys(self, input_keys):
        """
        Find all input and output keys in the chain set, determine which input keys are required, and validate the input keys.
        Required keys do not appear as output keys in any other chain.
        """

        # Find all input and output keys
        input_keys_set = set(
            input_variable for link in self.links for input_variable in link.input_keys
        )
        output_keys_set = set(link.output_key for link in self.links)
        required_keys = input_keys_set - output_keys_set
        logger.debug(f"Input keys: {input_keys_set}" f"Output keys: {output_keys_set}")

        # For more than one link, we do not require a state key, as one is always provided by the previous link
        if len(self.links) > 1:
            required_keys.discard("state")

        # Validate input keys
        if input_keys is None:
            raise ValueError(f"The following input keys are required: {required_keys}")
        elif not isinstance(input_keys, Collection):
            raise TypeError("Input_keys must be a Collection.")

        # Check if all required keys are present
        if not required_keys.issubset(input_keys):
            raise ValueError(
                f"The following input keys are required: {required_keys}."
                f" Missing: {required_keys - set(input_keys)}"
            )

    @abstractmethod
    def run(self, input_dict):
        """
        Run the Links in the correct order based on the dependencies in the graph.

        :param input_dict: Variables for the template.
        :return: Dictionary containing all the input and output variables.
        """
        raise NotImplementedError(
            "Base class <BaseChain> `run` method must be implemented."
        )


###################
# Implementations #
###################


class Chain(BaseChain):
    """
    Chain class that takes a list or set of Link objects, determines the order in which to run them, and does so.
    """

    def __init__(self, links: Collection[Link], llm: LLM = None):
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
        if self.llm is None and any(link.model is None for link in self.links):
            raise ValueError("An LLM model must be provided.")

        # Validate the input keys
        self._find_and_validate_keys(input_dict)

        # Initialize the output dictionary with the input dictionary, which will be passed to the first chain
        output_dict = input_dict

        # Iterate through Links in topological order
        for link in self._topological_sort():
            # Run the Link with Chain's LLM model if the Link doesn't have one
            if link.model is None:
                output_dict.update(link.run(input_dict=output_dict, llm=self.llm))
            else:
                output_dict.update(link.run(input_dict=output_dict))

        return output_dict

    def _topological_sort(self):
        """
        Perform topological sorting of the Links based on the dependencies in the graph.

        We use depth-first search to find the topological order of the Links.
        If a cycle is detected, an error is raised. We'll use a set to keep
        track of visited nodes and a stack to keep track of the current path.
        The stack represents the current path of the DFS traversal. If we
        revisit a node in the stack, we have a cycle.
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
            nonlocal visited, stack, sorted_links  # nonlocal keyword is used to work with variables inside nested functions

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

                # Add the link to the sorted list
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
        if self.llm is None and any(link.model is None for link in self.links):
            raise ValueError("An LLM model must be provided.")

        # Validate the input keys
        self._find_and_validate_keys(input_dict)

        # Initialize the output dictionary with the input dictionary, which will be passed to the first chain
        state_dict = input_dict

        # Iterate through Links in order
        for link in self.links:
            # Keep track of state keys
            incoming_state_keys = set(state_dict.keys())

            # Check if state_dict contains all the input keys for the Link
            if not all(input_key in state_dict for input_key in link.input_keys):
                raise ValueError(
                    f"Input keys {link.input_keys} not found in state_dict."
                )

            # Run the Link with Chain's LLM model if the Link doesn't have one
            if link.model is None:
                response_dict = link.run(input_dict=state_dict, llm=self.llm)
            else:
                response_dict = link.run(input_dict=state_dict)

            # Update the state_dict with the output of the Link
            state_dict.update(response_dict)

            # New state keys
            outgoing_state_keys = set(state_dict.keys())
            # State keys that were added by the Link
            new_state_keys = outgoing_state_keys - incoming_state_keys
            # State keys removed by the Link
            removed_state_keys = incoming_state_keys - outgoing_state_keys

            logger.info(
                f"🔑 State keys: {list(state_dict.keys())} (added: {list(new_state_keys)}, removed: {list(removed_state_keys)})"
            )

        return state_dict


if __name__ == "__main__":
    # Instantiate an LLM model
    claude = ClaudeInstant(source="test")

    # Example chains
    link_a = Link(
        template=StringTemplate(
            "What's the opposite of {{A}}? Reply with one word.", ["A"]
        ),
        name="opposite",
        output_key="X",
    )
    link_b = Link(
        template=StringTemplate(
            "What's a superlative version of {{B}}. Reply with one word.",
            ["B"],
        ),
        name="superlative",
        output_key="Y",
    )
    link_c = Link(
        template=StringTemplate(
            "Write a sarcastic poem using this as inspiration: '{{X}}' and '{{Y}}'",
            ["X", "Y"],
        ),
        name="sarcastic_poem",
        output_key="output",
    )

    # Combine the chains into a set
    chain = Chain(llm=claude, links=[link_a, link_b, link_c])

    # Run the chain set
    result = chain(input_dict={"A": "calm", "B": "a storm"})

    # print(result["output"])
    #
    # # Example ConcatLink
    # def concatenate(x, y):
    #     return f"{x} {y}"
    #
    # concat_link = TransformationLink(
    #     input_keys=["A", "B"], function=concatenate, output_key="C"
    # )
    # concat_link2 = TransformationLink(input_keys=["C", "D"], function=concatenate)
    # concat_link3 = TransformationLink(input_keys=["X", "Y"], function=concatenate)
    #
    # # Example LinearChain
    # linear_chain = LinearChain(llm=claude, links=[concat_link, concat_link2])
    # result = linear_chain(input_dict={"A": "I am", "B": "a calm", "D": "hurricane"})
    #
    # # Let's try another one
    # linear_chain = LinearChain(llm=claude, links=[link_a, link_b, concat_link3])
    # result_linear = linear_chain(
    #     input_dict={"A": "I am", "B": "a calm", "D": "hurricane"}
    # )
    #
    # # Let's see what error a BaseLink run method raises
    # try:
    #     base_link = BaseLink(input_keys=["A"], output_key="B")
    #     base_link.run(input_dict={"A": "test"})
    # except Exception as e:
    #     print('Gave this error:', e)
