import pytest

from fence import TOMLParser
from fence.links import Link
from fence.models.base import LLM
from fence.utils.shortcuts import create_string_link, create_toml_link


class MockLLM(LLM):
    """
    A mock implementation of the LLM class for testing purposes.
    """

    def __init__(self, return_string: str = None):
        self.model_id = "mock-llm"
        self.model_name = "Mock LLM"
        self.inference_type = "test"
        self.metric_prefix = "mock"
        self.return_string = return_string

    def invoke(self, *args, **kwargs):
        """
        Mock the invocation of the LLM, returning a static TOML-formatted string.
        """
        return self.return_string


@pytest.fixture
def mock_model_toml():
    """
    Provide a fixture for the mock LLM model.
    """
    return MockLLM(
        return_string="""```toml\n[[ingredients]]\nname = \"Flour\"\nquantity = \"2 cups\"```"""
    )


@pytest.fixture
def mock_model_string():
    """
    Provide a fixture for the mock LLM model.
    """
    return MockLLM(return_string="blue")


def test_create_toml_link(mock_model_toml):
    """
    Test the create_toml_link utility function.

    This test verifies that the link object is correctly created and configured, and that
    the run method produces the expected output.
    """
    # Define the input messages
    user_message = "Create an ingredient list for {recipe_name}"
    assistant_message = "```toml"
    system_message = """You create ingredient lists in a TOML format. Here's an example:
[[ingredients]]
name = "Flour"
quantity = "2 cups"

[[ingredients]]
name = "Sugar"
quantity = "1 cup"

[[ingredients]]
name = "Eggs"
quantity = "2"

[[ingredients]]
name = "Milk"
quantity = "1.5 cups"""

    # Create the link object
    link = create_toml_link(
        model=mock_model_toml,
        user_message=user_message,
        assistant_message=assistant_message,
        system_message=system_message,
    )

    # Assert the link object is correctly configured
    assert isinstance(link, Link)
    assert link.name == "toml_link"
    assert isinstance(link.parser, TOMLParser)

    # Run the link and verify the output
    result = link.run(recipe_name="chocolate cake")
    assert "state" in result
    assert result["state"] == {"ingredients": [{"name": "Flour", "quantity": "2 cups"}]}


def test_create_string_link(mock_model_string):
    """
    Test the create_string_link utility function.

    This test verifies that the link object is correctly created and configured, and that
    the run method produces the expected output.
    """
    # Define the input messages
    user_message = "The sky is"
    system_message = """Respond with a color"""

    # Create the link object
    link = create_string_link(
        model=mock_model_string,
        user_message=user_message,
        system_message=system_message,
    )

    # Assert the link object is correctly configured
    assert isinstance(link, Link)
    assert link.name == "string_link"
    assert link.parser is None

    # Run the link and verify the output
    result = link.run()
    assert "state" in result
    assert result["state"] == "blue"
