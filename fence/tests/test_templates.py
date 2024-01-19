import os
import tempfile

import pytest

from fence.demo.lib.llm.templates import PromptTemplate


def file_opener():
    return "test"


def test_prompt_template_initialization():
    """
    Test PromptTemplate initialization.
    """
    template = PromptTemplate("{{A}}", ["A"])
    assert template.source == "{{A}}"
    assert template.input_variables == ["A"]
    assert template.separator == " "


def test_prompt_template_call_alias_for_render():
    """
    Test the __call__ method as an alias for render.
    """
    template = PromptTemplate("{{A}}", ["A"])
    rendered = template(A="test")
    assert rendered == "test"


def test_prompt_template_addition_of_same_type():
    """
    Test addition of two PromptTemplate instances.
    """
    template1 = PromptTemplate("{{A}}", ["A"])
    template2 = PromptTemplate("{{B}}", ["B"])
    merged = template1 + template2
    assert merged.source == "{{A}} {{B}}"
    assert set(merged.input_variables) == {"A", "B"}


def test_prompt_template_addition_raises_error_for_different_type():
    """
    Test that adding a non-PromptTemplate object raises TypeError.
    """
    template = PromptTemplate("{{A}}", ["A"])
    with pytest.raises(TypeError):
        _ = template + "not a PromptTemplate"


def test_prompt_template_render_with_all_variables_provided():
    """
    Test rendering a template with all variables provided.
    """
    template = PromptTemplate("{{A}} {{B}}", ["A", "B"])
    rendered = template.render(A="test", B="123")
    assert rendered == "test 123"


def test_prompt_template_render_raises_error_for_missing_variables():
    """
    Test that rendering a template with missing variables raises ValueError.
    """
    template = PromptTemplate("{{A}} {{B}}", ["A", "B"])
    with pytest.raises(ValueError):
        _ = template.render(A="test")


def test_prompt_template_equality():
    """
    Test equality of PromptTemplate instances.
    """
    template1 = PromptTemplate("{{A}}", ["A"])
    template2 = PromptTemplate("{{A}}", ["A"])
    assert template1 == template2


def test_prompt_template_copy():
    """
    Test copying a PromptTemplate instance.
    """
    template = PromptTemplate("{{A}}", ["A"])
    copy = template.copy()
    assert template == copy
    assert template is not copy


@pytest.fixture
def temp_template_file():
    """
    Fixture to create a temporary file with a template.
    """
    template_content = "Hello, {{name}}! How are you today?"
    temp_file = tempfile.NamedTemporaryFile(mode="w+", delete=False)
    temp_file.write(template_content)
    temp_file.close()
    yield temp_file.name
    os.unlink(temp_file.name)  # Remove the temporary file after the test


def test_from_file(temp_template_file):
    """
    Test the from_file method to create a PromptTemplate from a file.
    """
    template_path = temp_template_file
    prompt_template = PromptTemplate.from_file(template_path)

    # Assert that the loaded template is correct
    assert prompt_template.source == "Hello, {{name}}! How are you today?"
    assert prompt_template.input_variables == ["name"]
