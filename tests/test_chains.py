from unittest.mock import Mock

import pytest

from fence.chains import BaseChain, Chain, LinearChain
from fence.links import BaseLink, Link
from fence.models.base import LLM
from fence.templates.string import StringTemplate


@pytest.fixture
def llm():
    """
    This fixture creates a mock LLM object that returns a fixed string when called.
    The purpose of this fixture is to provide a mock LLM for testing purposes.
    :return: Mock LLM object
    """
    mock_llm = Mock(LLM)
    mock_llm.return_value = "mocked response"
    return mock_llm


def test_base_chain_run_method_raises_error(llm):
    """
    Test case for the run method of the BaseChain class.
    This test case checks if the run method of the BaseChain class raises a TypeError.
    The BaseChain class is an abstract base class and its run method needs to be implemented in any derived class.
    """
    with pytest.raises(TypeError):
        base_chain = BaseChain(links=[Mock(BaseLink)], model=llm)
        base_chain.run(input_dict={"A": "test"})


def test_chain_run_method_without_llm_raises_error():
    """
    Test case for the run method of the Chain class without providing an LLM.
    This test checks if the run method raises a ValueError when no LLM is provided.
    """
    link = Link(template=StringTemplate("{{A}}"), output_key="B")
    chain = Chain(links=[link])
    with pytest.raises(ValueError):
        chain.run(input_dict={"A": "test"})


def test_linear_chain_run_method_without_llm_raises_error():
    """
    Test case for the run method of the LinearChain class without providing an LLM.
    This test checks if the run method raises a ValueError when no LLM is provided.
    """
    link = Link(template=StringTemplate("{{A}}"), output_key="B")
    linear_chain = LinearChain(links=[link])
    with pytest.raises(ValueError):
        linear_chain.run(input_dict={"A": "test"})
