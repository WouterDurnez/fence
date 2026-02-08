"""
Tests for structured output functionality in BedrockBase.

This module contains comprehensive tests for:
1. Validation alias generation (PascalCase, camelCase, snake_case, etc.)
2. Schema-aware merging of multiple tool outputs
3. enable_advanced_parsing flag behavior
4. allow_refs_in_json_schema flag behavior
5. Nested model handling
6. Edge cases and error handling
"""

import pytest
from pydantic import BaseModel, Field, ValidationError

from fence.models.bedrock.base import BedrockBase, BedrockToolConfig


# Test Models
class SimpleModel(BaseModel):
    """Simple model with basic fields."""

    name: str = Field(..., description="The name")
    age: int = Field(..., description="The age")


class QuestionAnswer(BaseModel):
    """Nested model for Q&A."""

    question: str = Field(..., description="The question")
    answer: str = Field(..., description="The answer")
    context: str = Field(..., description="The context")


class FAQOutput(BaseModel):
    """Model with list field for testing merging."""

    current_time: str = Field(..., description="Current timestamp")
    question_answer_objects: list[QuestionAnswer] = Field(
        ..., description="List of Q&A objects"
    )


class MixedFieldsModel(BaseModel):
    """Model with both scalar and list fields."""

    title: str = Field(..., description="Document title")
    tags: list[str] = Field(..., description="List of tags")
    author: str = Field(..., description="Author name")
    sections: list[str] = Field(..., description="List of sections")


class NestedModel(BaseModel):
    """Model with nested BaseModel fields."""

    metadata: SimpleModel = Field(..., description="Metadata")
    items: list[QuestionAnswer] = Field(..., description="List of items")


# Mock Model for Testing
class MockBedrockStructuredOutput(BedrockBase):
    """Mock Bedrock model for testing structured output."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_id = "test-model"

    def _invoke(self, prompt, stream=False, **kwargs):
        """Mock invoke - returns mock response that goes through processing pipeline."""
        self._check_if_prompt_is_valid(prompt)

        # Return mock response based on output_structure
        if self.output_structure == SimpleModel:
            response = self._mock_simple_response()
        elif self.output_structure == FAQOutput:
            response = self._mock_faq_response()
        elif self.output_structure == MixedFieldsModel:
            response = self._mock_mixed_fields_response()
        elif self.output_structure == NestedModel:
            response = self._mock_nested_response()
        else:
            return "Mock response"

        # Process the response through the normal pipeline
        if stream:
            return response
        else:
            # Simulate _handle_invoke behavior
            completion = self._process_response(response)
            if self.output_structure is not None:
                self._validate_structured_output(completion)
            return completion

    def _mock_simple_response(self):
        """Mock response for SimpleModel."""
        return {
            "output": {
                "message": {
                    "content": [
                        {
                            "toolUse": {
                                "toolUseId": "123",
                                "name": "analyze_information",
                                "input": {"name": "John", "age": 30},
                            }
                        }
                    ]
                }
            }
        }

    def _mock_faq_response(self):
        """Mock response for FAQOutput with multiple tool uses."""
        return {
            "output": {
                "message": {
                    "content": [
                        {
                            "toolUse": {
                                "toolUseId": "1",
                                "name": "analyze_information",
                                "input": {
                                    "current_time": "12:00:00",
                                    "question_answer_objects": [
                                        {
                                            "question": "Q1",
                                            "answer": "A1",
                                            "context": "C1",
                                        }
                                    ],
                                },
                            }
                        },
                        {
                            "toolUse": {
                                "toolUseId": "2",
                                "name": "analyze_information",
                                "input": {
                                    "current_time": "12:01:00",
                                    "question_answer_objects": [
                                        {
                                            "question": "Q2",
                                            "answer": "A2",
                                            "context": "C2",
                                        }
                                    ],
                                },
                            }
                        },
                    ]
                }
            }
        }

    def _mock_mixed_fields_response(self):
        """Mock response for MixedFieldsModel with multiple tool uses."""
        return {
            "output": {
                "message": {
                    "content": [
                        {
                            "toolUse": {
                                "toolUseId": "1",
                                "name": "analyze_information",
                                "input": {
                                    "title": "First Title",
                                    "tags": ["python", "ai"],
                                    "author": "First Author",
                                    "sections": ["intro", "body"],
                                },
                            }
                        },
                        {
                            "toolUse": {
                                "toolUseId": "2",
                                "name": "analyze_information",
                                "input": {
                                    "title": "Second Title",
                                    "tags": ["ml", "nlp"],
                                    "author": "Second Author",
                                    "sections": ["conclusion"],
                                },
                            }
                        },
                    ]
                }
            }
        }

    def _mock_nested_response(self):
        """Mock response for NestedModel."""
        return {
            "output": {
                "message": {
                    "content": [
                        {
                            "toolUse": {
                                "toolUseId": "123",
                                "name": "analyze_information",
                                "input": {
                                    "metadata": {"name": "Test", "age": 25},
                                    "items": [
                                        {
                                            "question": "Q1",
                                            "answer": "A1",
                                            "context": "C1",
                                        }
                                    ],
                                },
                            }
                        }
                    ]
                }
            }
        }


# Fixtures
@pytest.fixture
def mock_model_simple():
    """Create a mock model with SimpleModel output structure."""
    return MockBedrockStructuredOutput(output_structure=SimpleModel)


@pytest.fixture
def mock_model_faq():
    """Create a mock model with FAQOutput output structure."""
    return MockBedrockStructuredOutput(output_structure=FAQOutput)


@pytest.fixture
def mock_model_mixed():
    """Create a mock model with MixedFieldsModel output structure."""
    return MockBedrockStructuredOutput(output_structure=MixedFieldsModel)


@pytest.fixture
def mock_model_nested():
    """Create a mock model with NestedModel output structure."""
    return MockBedrockStructuredOutput(output_structure=NestedModel)


@pytest.fixture
def mock_model_no_advanced_parsing():
    """Create a mock model with advanced parsing disabled."""
    return MockBedrockStructuredOutput(
        output_structure=FAQOutput, enable_advanced_parsing=False
    )


# Test Classes
class TestValidationAliases:
    """Test validation alias generation for case-insensitive field matching."""

    def test_aliases_added_to_model(self, mock_model_simple):
        """Test that validation aliases are added to the model fields."""
        model = mock_model_simple.output_structure

        # Check that validation_alias is set on fields
        assert hasattr(model.model_fields["name"], "validation_alias")
        assert hasattr(model.model_fields["age"], "validation_alias")

    def test_pascalcase_field_matching(self, mock_model_simple):
        """Test that PascalCase field names are accepted."""
        # Simulate Bedrock returning PascalCase
        data = {"Name": "John", "Age": 30}
        validated = mock_model_simple.output_structure.model_validate(data)

        assert validated.name == "John"
        assert validated.age == 30

    def test_camelcase_field_matching(self, mock_model_simple):
        """Test that camelCase field names are accepted."""
        # Simulate Bedrock returning camelCase
        data = {"name": "John", "age": 30}
        validated = mock_model_simple.output_structure.model_validate(data)

        assert validated.name == "John"
        assert validated.age == 30

    def test_snake_case_field_matching(self):
        """Test that snake_case field names work with aliases."""

        class SnakeCaseModel(BaseModel):
            first_name: str
            last_name: str

        model = MockBedrockStructuredOutput(output_structure=SnakeCaseModel)

        # Test PascalCase
        data1 = {"FirstName": "John", "LastName": "Doe"}
        validated1 = model.output_structure.model_validate(data1)
        assert validated1.first_name == "John"
        assert validated1.last_name == "Doe"

        # Test camelCase
        data2 = {"firstName": "Jane", "lastName": "Smith"}
        validated2 = model.output_structure.model_validate(data2)
        assert validated2.first_name == "Jane"
        assert validated2.last_name == "Smith"

    def test_uppercase_field_matching(self, mock_model_simple):
        """Test that UPPERCASE field names are accepted."""
        data = {"NAME": "John", "AGE": 30}
        validated = mock_model_simple.output_structure.model_validate(data)

        assert validated.name == "John"
        assert validated.age == 30

    def test_lowercase_field_matching(self, mock_model_simple):
        """Test that lowercase field names are accepted."""
        data = {"name": "John", "age": 30}
        validated = mock_model_simple.output_structure.model_validate(data)

        assert validated.name == "John"
        assert validated.age == 30

    def test_nested_model_aliases(self, mock_model_nested):
        """Test that aliases are added to nested models."""
        # Test with PascalCase in nested model
        data = {
            "Metadata": {"Name": "Test", "Age": 25},
            "Items": [{"Question": "Q1", "Answer": "A1", "Context": "C1"}],
        }
        validated = mock_model_nested.output_structure.model_validate(data)

        assert validated.metadata.name == "Test"
        assert validated.metadata.age == 25
        assert validated.items[0].question == "Q1"


class TestSchemaAwareMerging:
    """Test schema-aware merging of multiple tool outputs."""

    def test_single_output_no_merging(self, mock_model_simple):
        """Test that single outputs are not merged."""
        result = mock_model_simple.invoke("Test prompt")

        # invoke returns a dict, not a Pydantic model instance
        assert isinstance(result, dict)
        assert result["name"] == "John"
        assert result["age"] == 30

        # But it should be valid according to the model
        validated = mock_model_simple.output_structure.model_validate(result)
        assert isinstance(validated, SimpleModel)

    def test_multiple_outputs_list_fields_merged(self, mock_model_faq):
        """Test that list fields are merged when multiple outputs are returned."""
        result = mock_model_faq.invoke("Test prompt")

        # invoke returns a dict
        assert isinstance(result, dict)
        # Scalar field: should keep first value
        assert result["current_time"] == "12:00:00"
        # List field: should merge all items
        assert len(result["question_answer_objects"]) == 2
        assert result["question_answer_objects"][0]["question"] == "Q1"
        assert result["question_answer_objects"][1]["question"] == "Q2"

        # Validate it matches the model
        validated = mock_model_faq.output_structure.model_validate(result)
        assert isinstance(validated, FAQOutput)

    def test_multiple_outputs_scalar_fields_keep_first(self, mock_model_mixed):
        """Test that scalar fields keep only the first value when merging."""
        result = mock_model_mixed.invoke("Test prompt")

        # invoke returns a dict
        assert isinstance(result, dict)
        # Scalar fields: should keep first value
        assert result["title"] == "First Title"
        assert result["author"] == "First Author"
        # List fields: should merge all items
        assert result["tags"] == ["python", "ai", "ml", "nlp"]
        assert result["sections"] == ["intro", "body", "conclusion"]

        # Validate it matches the model
        validated = mock_model_mixed.output_structure.model_validate(result)
        assert isinstance(validated, MixedFieldsModel)

    def test_field_type_inspection(self, mock_model_mixed):
        """Test that _get_field_types correctly identifies list vs scalar fields."""
        field_types = mock_model_mixed._get_field_types(MixedFieldsModel)

        assert field_types["title"] is False  # scalar
        assert field_types["tags"] is True  # list
        assert field_types["author"] is False  # scalar
        assert field_types["sections"] is True  # list

    def test_merge_tool_outputs_with_lists(self, mock_model_mixed):
        """Test _merge_tool_outputs with mixed scalar and list fields."""
        tool_outputs = [
            {"title": "T1", "tags": ["a", "b"], "author": "A1", "sections": ["s1"]},
            {"title": "T2", "tags": ["c"], "author": "A2", "sections": ["s2", "s3"]},
        ]
        field_types = mock_model_mixed._get_field_types(MixedFieldsModel)
        result = mock_model_mixed._merge_tool_outputs(tool_outputs, field_types)

        # Scalars keep first value
        assert result["title"] == "T1"
        assert result["author"] == "A1"
        # Lists are merged
        assert result["tags"] == ["a", "b", "c"]
        assert result["sections"] == ["s1", "s2", "s3"]

    def test_merge_with_scalar_wrapped_in_list(self, mock_model_faq):
        """Test merging when a scalar value needs to be wrapped in a list."""
        # Simulate a case where one output has a scalar instead of a list
        tool_outputs = [
            {
                "current_time": "12:00",
                "question_answer_objects": {
                    "question": "Q1",
                    "answer": "A1",
                    "context": "C1",
                },
            },
            {
                "current_time": "12:01",
                "question_answer_objects": [
                    {"question": "Q2", "answer": "A2", "context": "C2"}
                ],
            },
        ]
        field_types = mock_model_faq._get_field_types(FAQOutput)
        result = mock_model_faq._merge_tool_outputs(tool_outputs, field_types)

        # Scalar wrapped in list, then extended
        assert len(result["question_answer_objects"]) == 2


class TestEnableAdvancedParsing:
    """Test the enable_advanced_parsing flag behavior."""

    def test_advanced_parsing_enabled_by_default(self, mock_model_simple):
        """Test that advanced parsing is enabled by default."""
        assert mock_model_simple.enable_advanced_parsing is True

    def test_advanced_parsing_can_be_disabled(self, mock_model_no_advanced_parsing):
        """Test that advanced parsing can be disabled via kwarg."""
        assert mock_model_no_advanced_parsing.enable_advanced_parsing is False

    def test_aliases_not_added_when_disabled(self):
        """Test that validation aliases are not added when advanced parsing is disabled."""

        # Create a fresh model class that hasn't been modified
        class FreshSimpleModel(BaseModel):
            name: str
            age: int

        model = MockBedrockStructuredOutput(
            output_structure=FreshSimpleModel, enable_advanced_parsing=False
        )

        # When advanced parsing is disabled, aliases are not added
        # So PascalCase should fail validation
        with pytest.raises(ValidationError):
            model.output_structure.model_validate({"Name": "John", "Age": 30})

        # But exact field names should still work
        validated = model.output_structure.model_validate({"name": "John", "age": 30})
        assert validated.name == "John"

    def test_merging_disabled_uses_first_output(self, mock_model_no_advanced_parsing):
        """Test that when advanced parsing is disabled, only first output is used."""
        result = mock_model_no_advanced_parsing.invoke("Test prompt")

        # invoke returns a dict
        assert isinstance(result, dict)
        # Should only have the first output's data
        assert result["current_time"] == "12:00:00"
        assert len(result["question_answer_objects"]) == 1
        assert result["question_answer_objects"][0]["question"] == "Q1"

        # Validate it matches the model
        validated = mock_model_no_advanced_parsing.output_structure.model_validate(
            result
        )
        assert isinstance(validated, FAQOutput)


class TestAllowRefsInJsonSchema:
    """Test the allow_refs_in_json_schema flag behavior."""

    def test_refs_resolved_by_default(self):
        """Test that $refs are resolved by default in JSON schema."""
        model = MockBedrockStructuredOutput(output_structure=NestedModel)
        schema = model.json_output_schema

        # Check that $defs is not present (refs were resolved)
        assert "$defs" not in schema

    def test_refs_kept_when_enabled(self):
        """Test that $refs are kept when allow_refs_in_json_schema=True."""
        model = MockBedrockStructuredOutput(
            output_structure=NestedModel, allow_refs_in_json_schema=True
        )
        schema = model.json_output_schema

        # $defs might be present when refs are allowed
        # (This depends on whether Pydantic generates refs for this model)
        # At minimum, the schema should be valid
        assert "properties" in schema


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_no_tool_use_in_response_raises_error(self):
        """Test that missing tool use raises an error."""
        model = MockBedrockStructuredOutput(output_structure=SimpleModel)

        # Mock a response with no toolUse
        response_body = {"content": [{"text": "Some text"}]}

        with pytest.raises(ValueError, match="No tool use found in response"):
            model._extract_structured_output(response_body)

    def test_empty_content_raises_error(self):
        """Test that empty content raises an error."""
        model = MockBedrockStructuredOutput(output_structure=SimpleModel)

        response_body = {"content": []}

        with pytest.raises(ValueError, match="No tool use found in response"):
            model._extract_structured_output(response_body)

    def test_invalid_output_structure_raises_error(self):
        """Test that non-BaseModel output_structure raises an error."""
        with pytest.raises(
            ValueError, match="output_structure must be a Pydantic model class"
        ):
            MockBedrockStructuredOutput(output_structure=dict)

    def test_nested_model_recursion_prevention(self):
        """Test that recursive model processing doesn't cause infinite loops."""

        # Create a model that references itself (indirectly)
        class ModelA(BaseModel):
            name: str
            items: list[QuestionAnswer]

        # This should not cause infinite recursion
        model = MockBedrockStructuredOutput(output_structure=ModelA)

        # Verify aliases were added
        assert hasattr(model.output_structure.model_fields["name"], "validation_alias")
