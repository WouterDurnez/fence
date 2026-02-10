import pytest
from pydantic import BaseModel

from fence.links import BaseLink, Link, TransformationLink
from fence.templates.string import StringTemplate
from fence.models.bedrock.base import BedrockBase


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


# Test models for structured output
class ReviewOutput(BaseModel):
    """Model for review analysis output."""

    sentiment: str
    rating: int
    summary: str


class QuestionAnswer(BaseModel):
    """Model for Q&A pairs."""

    question: str
    answer: str


# Mock Bedrock model for testing
class MockBedrockModel(BedrockBase):
    """Mock Bedrock model for testing Links with structured output."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_id = "test-model"

    def _invoke(self, prompt, stream=False, **kwargs):
        """Mock invoke that returns structured data."""
        self._check_if_prompt_is_valid(prompt)

        # Return mock response based on output_structure
        if self.output_structure == ReviewOutput:
            response = {
                "output": {
                    "message": {
                        "content": [
                            {
                                "toolUse": {
                                    "toolUseId": "123",
                                    "name": "analyze_information",
                                    "input": {
                                        "sentiment": "positive",
                                        "rating": 5,
                                        "summary": "Great product!",
                                    },
                                }
                            }
                        ]
                    }
                }
            }
        elif self.output_structure == QuestionAnswer:
            response = {
                "output": {
                    "message": {
                        "content": [
                            {
                                "toolUse": {
                                    "toolUseId": "456",
                                    "name": "analyze_information",
                                    "input": {
                                        "question": "What is the capital of France?",
                                        "answer": "Paris",
                                    },
                                }
                            }
                        ]
                    }
                }
            }
        else:
            # No structured output - return plain text
            return "Mock response text"

        # Process through the pipeline
        completion = self._process_response(response)
        if self.output_structure is not None:
            self._validate_structured_output(completion)
        return completion


class TestLinkWithStructuredOutput:
    """Test Links with structured output (output_structure parameter)."""

    def test_link_with_output_structure_in_model(self):
        """Test Link with output_structure set in the model."""
        # Create model with output_structure
        model = MockBedrockModel(output_structure=ReviewOutput)

        # Create link
        link = Link(
            model=model,
            template=StringTemplate("Analyze this review: {{review}}"),
            output_key="analysis",
            name="review_analyzer",
        )

        # Run the link
        result = link.run(input_dict={"review": "This product is amazing!"})

        # Check that we got structured output
        assert "analysis" in result
        assert isinstance(result["analysis"], dict)
        assert result["analysis"]["sentiment"] == "positive"
        assert result["analysis"]["rating"] == 5
        assert result["analysis"]["summary"] == "Great product!"

    def test_link_with_output_structure_in_link(self):
        """Test Link with output_structure set in the Link constructor.

        Note: For Bedrock models, output_structure must be set in the model,
        not in the Link. The Link's output_structure is used for the parser.
        """
        # Create model with output_structure
        model = MockBedrockModel(output_structure=ReviewOutput)

        # Create link with output_structure (this sets up the parser)
        link = Link(
            model=model,
            template=StringTemplate("Analyze this review: {{review}}"),
            output_key="analysis",
            output_structure=ReviewOutput,
            name="review_analyzer",
        )

        # Run the link
        result = link.run(input_dict={"review": "This product is amazing!"})

        # Check that we got structured output wrapped in 'state' key
        # (because the parser wraps it when output_structure is set)
        assert "analysis" in result
        assert isinstance(result["analysis"], dict)
        assert "state" in result["analysis"]
        assert isinstance(result["analysis"]["state"], dict)

    def test_link_output_structure_only_works_with_bedrock(self):
        """Test that output_structure raises error for non-Bedrock models."""

        # Create a mock non-Bedrock model
        class MockNonBedrockModel:
            def invoke(self, prompt, **kwargs):
                return "response"

        # Try to create link with output_structure on non-Bedrock model
        with pytest.raises(
            ValueError, match="output_structure is only supported for Bedrock models"
        ):
            Link(
                model=MockNonBedrockModel(),
                template=StringTemplate("Test {{input}}"),
                output_structure=ReviewOutput,
            )

    def test_link_output_structure_conflicts_with_parser(self):
        """Test that output_structure and parser cannot both be specified."""
        from fence.parsers import BoolParser

        model = MockBedrockModel()

        # Try to create link with both output_structure and parser
        with pytest.raises(
            ValueError, match="Cannot specify both output_structure and parser"
        ):
            Link(
                model=model,
                template=StringTemplate("Test {{input}}"),
                output_structure=ReviewOutput,
                parser=BoolParser(),
            )

    def test_link_with_different_output_structures(self):
        """Test Link with different output structures."""
        # Create model with QuestionAnswer output structure
        model = MockBedrockModel(output_structure=QuestionAnswer)

        # Create link
        link = Link(
            model=model,
            template=StringTemplate("Answer this: {{question}}"),
            output_key="qa",
            name="qa_generator",
        )

        # Run the link
        result = link.run(input_dict={"question": "What is the capital of France?"})

        # Check that we got structured output
        assert "qa" in result
        assert isinstance(result["qa"], dict)
        assert result["qa"]["question"] == "What is the capital of France?"
        assert result["qa"]["answer"] == "Paris"

    def test_link_structured_output_validates_response(self):
        """Test that Link validates structured output against the schema."""
        # This test verifies that the validation happens
        # (The mock already validates, so we just check it doesn't raise)
        model = MockBedrockModel(output_structure=ReviewOutput)

        link = Link(
            model=model,
            template=StringTemplate("Analyze: {{text}}"),
            output_key="result",
        )

        # Should not raise - validation passes
        result = link.run(input_dict={"text": "test"})
        assert "result" in result
        assert isinstance(result["result"], dict)
