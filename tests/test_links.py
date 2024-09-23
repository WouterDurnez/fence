import pytest

from fence.links import BaseLink, Link, TransformationLink
from fence.templates.string import StringTemplate


def test_base_link_run_method_raises_error():
    """
    Test case for the run method of the BaseLink class.
    The BaseLink class is an abstract base class and its run method needs to be implemented in any derived class.
    The test case creates an instance of BaseLink and calls its run method, expecting a TypeError to be raised.
    :return: None
    """
    with pytest.raises(TypeError):
        base_link = BaseLink("B")
        base_link.run(input_dict={"A": "test"})


def test_transformation_link_run_method_returns_transformed_input():
    """
    Test case for the run method of the TransformationLink class.
    This test checks if the run method correctly transforms the input using the provided function.
    """
    transformation_link = TransformationLink(
        input_keys=["A"], output_key="B", function=lambda x: x["A"].upper()
    )
    result = transformation_link.run(input_dict={"A": "test"})
    assert result["state"] == "TEST"


def test_link_run_method_without_llm_raises_error():
    """
    Test case for the run method of the Link class without providing an LLM.
    This test checks if the run method raises a ValueError when no LLM is provided.
    """
    link = Link(template=StringTemplate("{{A}}"), output_key="B")
    with pytest.raises(ValueError):
        link.run(input_dict={"A": "test"})
