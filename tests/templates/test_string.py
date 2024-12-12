"""
String template tests.
"""

import logging
import os
import tempfile

import pytest

from fence.templates.string import StringTemplate


def file_opener():
    return "test"


def test_prompt_template_initialization():
    """
    Test PromptTemplate initialization.
    """
    template = StringTemplate("{A}")
    assert template.source == "{A}"
    assert template.input_variables == ["A"]
    assert template.separator == " "


def test_prompt_template_call_alias_for_render():
    """
    Test the __call__ method as an alias for render.
    """
    template = StringTemplate("{A}")
    rendered = template(A="test")
    assert rendered == "test"


def test_prompt_template_addition_of_same_type():
    """
    Test addition of two PromptTemplate instances.
    """
    template1 = StringTemplate("{A}")
    template2 = StringTemplate("{B}")
    merged = template1 + template2
    assert merged.source == "{A} {B}"
    assert set(merged.input_variables) == {"A", "B"}


def test_prompt_template_addition_raises_error_for_different_type():
    """
    Test that adding a non-PromptTemplate object raises TypeError.
    """
    template = StringTemplate("{A}")
    with pytest.raises(ValueError):
        _ = template + "not a PromptTemplate"


def test_prompt_template_render_with_all_variables_provided():
    """
    Test rendering a template with all variables provided.
    """
    template = StringTemplate("{A} {B}")
    rendered = template.render(A="test", B="123")
    assert rendered == "test 123"


def test_prompt_template_render_gives_warning_for_missing_variables(caplog):
    """
    Test that rendering a template with missing variables raises ValueError.
    """
    template = StringTemplate("{A} {B}")

    with caplog.at_level(logging.WARNING):
        _ = template.render(A="test")

    # Check if the warning about missing variables was logged
    assert any(
        "Possible missing variables" in message for message in caplog.text.splitlines()
    )


def test_prompt_template_render_with_superfluous_variables(caplog):
    """
    Test that rendering a template with superfluous variables logs a debug message.
    """
    template = StringTemplate("{A}")

    with caplog.at_level(logging.DEBUG):
        _ = template.render(A="test", B="123")

    # Check if the debug message about superfluous variables was logged
    assert any(
        "Superfluous variables" in message for message in caplog.text.splitlines()
    )


def test_prompt_template_render_nested_placeholder_attribute():
    """
    Test rendering a template with nested placeholder attributes.
    """
    template = StringTemplate("{A.B} {C}")
    rendered = template.render(C="test")
    assert rendered == "{A.B} test"
    assert set(template.input_variables) == {"A.B", "C"}


def test_prompt_template_equality():
    """
    Test equality of PromptTemplate instances.
    """
    template1 = StringTemplate("{A}")
    template2 = StringTemplate("{A}")
    assert template1 == template2


def test_prompt_template_copy():
    """
    Test copying a PromptTemplate instance.
    """
    template = StringTemplate("{A}")
    copy = template.copy()
    assert template == copy
    assert template is not copy


@pytest.fixture
def temp_template_file():
    """
    Fixture to create a temporary file with a template.
    """
    template_content = "Hello, {name}! How are you today?"
    temp_file = tempfile.NamedTemporaryFile(mode="w+", delete=False)
    temp_file.write(template_content)
    temp_file.close()
    yield temp_file.name
    os.unlink(temp_file.name)  # Remove the temporary file after the test
