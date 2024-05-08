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
    template = StringTemplate("{{A}}")
    assert template.source == "{{A}}"
    assert template.input_variables == ["A"]
    assert template.separator == " "


def test_prompt_template_call_alias_for_render():
    """
    Test the __call__ method as an alias for render.
    """
    template = StringTemplate("{{A}}")
    rendered = template(A="test")
    assert rendered == "test"


def test_prompt_template_addition_of_same_type():
    """
    Test addition of two PromptTemplate instances.
    """
    template1 = StringTemplate("{{A}}")
    template2 = StringTemplate("{{B}}")
    merged = template1 + template2
    assert merged.source == "{{A}} {{B}}"
    assert set(merged.input_variables) == {"A", "B"}


def test_prompt_template_addition_raises_error_for_different_type():
    """
    Test that adding a non-PromptTemplate object raises TypeError.
    """
    template = StringTemplate("{{A}}")
    with pytest.raises(ValueError):
        _ = template + "not a PromptTemplate"


def test_prompt_template_render_with_all_variables_provided():
    """
    Test rendering a template with all variables provided.
    """
    template = StringTemplate("{{A}} {{B}}")
    rendered = template.render(A="test", B="123")
    assert rendered == "test 123"


def test_prompt_template_render_raises_error_for_missing_variables():
    """
    Test that rendering a template with missing variables raises ValueError.
    """
    template = StringTemplate("{{A}} {{B}}")
    with pytest.raises(ValueError):
        _ = template.render(A="test")


def test_prompt_template_equality():
    """
    Test equality of PromptTemplate instances.
    """
    template1 = StringTemplate("{{A}}")
    template2 = StringTemplate("{{A}}")
    assert template1 == template2


def test_prompt_template_copy():
    """
    Test copying a PromptTemplate instance.
    """
    template = StringTemplate("{{A}}")
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


