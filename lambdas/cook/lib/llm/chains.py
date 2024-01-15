from collections import defaultdict
from typing import Iterable

from lambdas.cook.lib.llm.models import ClaudeInstantLLM
from lambdas.cook.lib.llm.templates import PromptTemplate


class Chain:
    """
    Chain class that takes an LLM model, and calls it using the prompt template. Base blocks for building more complex
    chains. The Chain class is callable, and will call the run method. The run method takes a dictionary of input
    variables for the template, and returns a dictionary with the output of the LLM model under the specified key.
    """

    def __init__(
        self,
        template: PromptTemplate,
        output_key: str = "output",
        llm=None,
        name: str = None,
    ):
        """
        Initialize the Chain object.

        :param name: Name of the chain.
        :param llm: An LLM model object.
        :param template: A PromptTemplate object.
        """
        self.name = name
        self.llm = llm
        self.template = template
        self.output_key = output_key

    def __call__(self, **kwargs):
        """
        Alias for the run method to allow calling an instance as a function.
        :param kwargs: Keyword arguments to be passed to the run method.
        :return: Output of the run method.
        """
        return self.run(**kwargs)

    def run(self, input_dict: dict, **kwargs):
        """
        Run the Chain.
        :param input_dict: Variables for the template.
        :param kwargs: Additional keyword arguments for the LLM model.
        :return:
        """
        # Check if an LLM model was provided
        if self.llm is None and kwargs.get("llm") is None:
            raise ValueError("An LLM model must be provided.")

        prompt = self.template(**input_dict)
        llm = kwargs.pop("llm", self.llm)
        return {self.output_key: llm(prompt, **kwargs)}

    def __str__(self):
        return f"Chain: {f'{self.name} ' if self.name else ''}<{self.template.input_variables}> -> <{self.output_key}>"

    def __repr__(self):
        return str(self)


class ChainSet:
    """
    ChainSet class that takes a list or set of Chain objects, determines the order in which to run them, and does so.
    """

    def __init__(self, chain_set: Iterable[Chain], llm=None):
        """
        Initialize the ChainSet object.

        :param chain_set: A set of Chain objects.
        """
        self.chain_set = chain_set
        self.graph = defaultdict(list)
        self.llm = llm

        # Build the graph
        self._build_graph()

    def _build_graph(self):
        """
        Get the input and output variables of each Chain in the ChainSet, and build a graph of dependencies.
        """
        for chain in self.chain_set:
            current_output_key = chain.output_key
            dependent_chains = [
                other_chain
                for other_chain in self.chain_set
                if current_output_key in other_chain.template.input_variables
            ]
            self.graph[chain].extend(dependent_chains)

    def run(self, input_dict):
        """
        Run the chains in the correct order based on the dependencies in the graph.

        :param input_dict: Variables for the template.
        :return: Dictionary containing the output of each chain.
        """

        # Check if an LLM model was provided
        if self.llm is None and any(chain.llm is None for chain in self.chain_set):
            raise ValueError("An LLM model must be provided.")

        # Validate the input keys
        self._find_and_validate_keys(input_dict)

        # Initialize the output dictionary with the input dictionary, which will be passed to the first chain
        output_dict = input_dict

        # Iterate through chains in topological order
        for chain in self._topological_sort():

            # Run the chain with Set's LLM model if the chain doesn't have one
            if chain.llm is None:
                output_dict.update(chain.run(input_dict=output_dict, llm=self.llm))
            else:
                output_dict.update(chain.run(input_dict=output_dict))

        return output_dict

    def _topological_sort(self):
        """
        Perform topological sorting of the chains based on the dependencies in the graph.
        """
        visited = set()
        stack = set()
        result = []

        def depth_first_search(chain):
            nonlocal visited, stack, result

            if chain in stack:
                # Cycle detected
                raise ValueError("Cycle detected in the dependency graph.")

            if chain not in visited:
                visited.add(chain)
                stack.add(chain)

                for dependent_chain in self.graph[chain]:
                    depth_first_search(dependent_chain)

                stack.remove(chain)
                result.insert(0, chain)

        for chain in self.chain_set:
            depth_first_search(chain)

        return result

    def _find_and_validate_keys(self, input_keys):
        """
        Find all input and output keys in the chain set, determine which input keys are required, and validate the input keys.
        Required keys do not appear as output keys in any other chain.
        """
        input_keys_set = set(
            input_variable
            for chain in self.chain_set
            for input_variable in chain.template.input_variables
        )
        output_keys = set(chain.output_key for chain in self.chain_set)
        required_keys = input_keys_set - output_keys

        # Validate input keys
        if input_keys is None:
            raise ValueError(f"The following input keys are required: {required_keys}")
        elif not isinstance(input_keys, Iterable):
            raise TypeError("Input_keys must be an iterable")

    def __call__(self, *args, **kwargs):
        """
        Alias for the run method to allow calling an instance as a function.
        :param kwargs: Keyword arguments to be passed to the run method.
        :return: Output of the run method.
        """
        return self.run(*args, **kwargs)


if __name__ == "__main__":
    claude = ClaudeInstantLLM(source="test")
    # template = PromptTemplate(
    #     template="Can you define and describe the following to me: {{input}}.",
    #     input_variables=["input"],
    # )
    # chain = Chain(llm=claude, template=template)
    #
    # print(chain.run(input_dict={"input": "Spanish inquisition"}))

    # Example chains
    chain_x = Chain(
        template=PromptTemplate(
            "What's the opposite of {{A}}? Reply with one word.", ["A"]
        ),
        output_key="X",
    )
    chain_a = Chain(
        template=PromptTemplate("Write a very short story about {{B}}. Two sentences max.", ["B"]),
        output_key="Y",
    )
    chain_b = Chain(
        template=PromptTemplate(
            "Write a sarcastic poem using this as inspiration: '{{X}}' and '{{Y}}'", ["X", "Y"]
        ),
        output_key="output",
    )


    chain_set = ChainSet(llm=claude, chain_set=[chain_x, chain_a, chain_b])

    result = chain_set(input_dict={"A": "calm", "B": "a hurricane"})

    print(result["output"])
