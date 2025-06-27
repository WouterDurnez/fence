"""
Tests for the models module.
"""

from decimal import Decimal

from fence.templates.models import ToolUseBlock


class TestToolUseBlock:
    """Test the ToolUseBlock model."""

    def test_convert_to_builtin_simple_decimal(self):
        """Test converting simple Decimal values to float."""
        input_data = {"score": Decimal("5.5"), "count": 10}

        tool_use = ToolUseBlock(toolUseId="test_id", name="test_tool", input=input_data)

        # Check that Decimal was converted to float
        assert isinstance(tool_use.input["score"], float)
        assert tool_use.input["score"] == 5.5
        # Check that non-Decimal values remain unchanged
        assert isinstance(tool_use.input["count"], int)
        assert tool_use.input["count"] == 10

    def test_convert_to_builtin_nested_dict(self):
        """Test converting Decimal values in nested dictionaries."""
        input_data = {
            "assessment": {
                "id": "test-id",
                "version": Decimal("15"),
                "selfReview": {"score": Decimal("4.5"), "status": "SUBMITTED"},
            }
        }

        tool_use = ToolUseBlock(toolUseId="test_id", name="test_tool", input=input_data)

        # Check that all Decimal values were converted to float
        assert isinstance(tool_use.input["assessment"]["version"], float)
        assert tool_use.input["assessment"]["version"] == 15.0
        assert isinstance(tool_use.input["assessment"]["selfReview"]["score"], float)
        assert tool_use.input["assessment"]["selfReview"]["score"] == 4.5
        # Check that non-Decimal values remain unchanged
        assert isinstance(tool_use.input["assessment"]["id"], str)
        assert tool_use.input["assessment"]["id"] == "test-id"

    def test_convert_to_builtin_nested_list(self):
        """Test converting Decimal values in nested lists."""
        input_data = {
            "competencyFeedbacks": [
                {"competencyId": "comp1", "score": Decimal("5")},
                {"competencyId": "comp2", "score": Decimal("3.5")},
                {"competencyId": "comp3", "score": Decimal("4")},
            ]
        }

        tool_use = ToolUseBlock(toolUseId="test_id", name="test_tool", input=input_data)

        # Check that all Decimal values in the list were converted to float
        for feedback in tool_use.input["competencyFeedbacks"]:
            assert isinstance(feedback["score"], float)

        assert tool_use.input["competencyFeedbacks"][0]["score"] == 5.0
        assert tool_use.input["competencyFeedbacks"][1]["score"] == 3.5
        assert tool_use.input["competencyFeedbacks"][2]["score"] == 4.0

    def test_convert_to_builtin_complex_nested(self):
        """Test converting Decimal values in complex nested structures."""
        input_data = {
            "assessment": {
                "id": "7573876f-4f95-47b9-b134-afc657107d8e",
                "version": Decimal("15"),
                "selfReview": {
                    "id": "d14e43c0-9888-4f3e-bd23-007b929bc91e",
                    "status": "SUBMITTED",
                    "competencyFeedbacks": [
                        {"competencyId": "comp1", "score": Decimal("5")},
                        {"competencyId": "comp2", "score": Decimal("4")},
                    ],
                },
                "managerReview": {
                    "competencyFeedbacks": [
                        {"competencyId": "comp1", "score": Decimal("1")},
                        {"competencyId": "comp2", "score": Decimal("2")},
                    ]
                },
            }
        }

        tool_use = ToolUseBlock(toolUseId="test_id", name="test_tool", input=input_data)

        # Check that all Decimal values were converted to float
        assert isinstance(tool_use.input["assessment"]["version"], float)
        assert tool_use.input["assessment"]["version"] == 15.0

        # Check selfReview scores
        for feedback in tool_use.input["assessment"]["selfReview"][
            "competencyFeedbacks"
        ]:
            assert isinstance(feedback["score"], float)
        assert (
            tool_use.input["assessment"]["selfReview"]["competencyFeedbacks"][0][
                "score"
            ]
            == 5.0
        )
        assert (
            tool_use.input["assessment"]["selfReview"]["competencyFeedbacks"][1][
                "score"
            ]
            == 4.0
        )

        # Check managerReview scores
        for feedback in tool_use.input["assessment"]["managerReview"][
            "competencyFeedbacks"
        ]:
            assert isinstance(feedback["score"], float)
        assert (
            tool_use.input["assessment"]["managerReview"]["competencyFeedbacks"][0][
                "score"
            ]
            == 1.0
        )
        assert (
            tool_use.input["assessment"]["managerReview"]["competencyFeedbacks"][1][
                "score"
            ]
            == 2.0
        )

    def test_convert_to_builtin_no_decimals(self):
        """Test that non-Decimal values remain unchanged."""
        input_data = {
            "string_value": "test",
            "int_value": 42,
            "float_value": 3.14,
            "bool_value": True,
            "list_value": [1, 2, 3],
            "dict_value": {"key": "value"},
        }

        tool_use = ToolUseBlock(toolUseId="test_id", name="test_tool", input=input_data)

        # Check that all values remain unchanged
        assert tool_use.input["string_value"] == "test"
        assert tool_use.input["int_value"] == 42
        assert tool_use.input["float_value"] == 3.14
        assert tool_use.input["bool_value"] is True
        assert tool_use.input["list_value"] == [1, 2, 3]
        assert tool_use.input["dict_value"] == {"key": "value"}
