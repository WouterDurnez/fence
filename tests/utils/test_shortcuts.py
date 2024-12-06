import pytest

from fence.links import Link
from fence.models.base import LLM
from fence.parsers import TOMLParser
from fence.utils.shortcuts import create_string_link, create_toml_link


class MockLLM(LLM):
    def __init__(self, return_string: str = None):
        self.model_id = "mock-llm"
        self.model_name = "Mock LLM"
        self.inference_type = "test"
        self.metric_prefix = "mock"
        self.return_string = return_string

    def invoke(self, *args, **kwargs):
        return self.return_string


@pytest.fixture(
    params=[
        {
            "model_type": "toml",
            "return_string": """```toml\n[[ingredients]]\nname = \"Flour\"\nquantity = \"2 cups\"```""",
            "user_message": "Create an ingredient list for {recipe_name}",
            "assistant_message": "```toml",
            "system_message": """You create ingredient lists in a TOML format. Here's an example:
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
quantity = "1.5 cups""",
            "expected_state": {
                "ingredients": [{"name": "Flour", "quantity": "2 cups"}]
            },
            "expected_name": "toml_link",
            "parser_class": TOMLParser,
        },
        {
            "model_type": "string",
            "return_string": "blue",
            "user_message": "The sky is",
            "system_message": "Respond with a color",
            "expected_state": "blue",
            "expected_name": "string_link",
            "parser_class": None,
        },
    ]
)
def mock_model(request):
    return {
        "model": MockLLM(return_string=request.param["return_string"]),
        **request.param,
    }


def test_link_creation(mock_model):
    """
    Parametrized test for link creation and running.
    Covers both TOML and string link scenarios.
    """
    if mock_model["model_type"] == "toml":
        link = create_toml_link(
            model=mock_model["model"],
            user_message=mock_model["user_message"],
            assistant_message=mock_model.get("assistant_message"),
            system_message=mock_model["system_message"],
        )
        result = link.run(recipe_name="chocolate cake")
    else:
        link = create_string_link(
            model=mock_model["model"],
            user_message=mock_model["user_message"],
            system_message=mock_model["system_message"],
        )
        result = link.run()

    # Common assertions
    assert isinstance(link, Link)
    assert link.name == mock_model["expected_name"]
    assert (
        isinstance(link.parser, mock_model["parser_class"])
        if mock_model["parser_class"]
        else link.parser is None
    )

    # State-specific assertion
    assert "state" in result
    assert result["state"] == mock_model["expected_state"]
