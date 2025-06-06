import tomllib

import pytest
from pydantic import BaseModel

from fence.parsers import (
    BoolParser,
    IntParser,
    PydanticParser,
    TOMLParser,
    TripleBacktickParser,
)


class Person(BaseModel):
    """Pydantic model for PydanticParser tests."""

    name: str
    age: int
    email: str


class Address(BaseModel):
    """Pydantic model with optional fields."""

    street: str
    city: str
    zipcode: str | None = None


def test_int_parser_with_valid_input():
    """
    Test IntParser with a valid integer input.
    """
    parser = IntParser()
    result = parser.parse("123")
    assert result == 123


def test_int_parser_with_invalid_input():
    """
    Test IntParser with an invalid (non-integer) input.
    """
    parser = IntParser()
    with pytest.raises(ValueError):
        parser.parse("abc")


def test_bool_parser_with_valid_input():
    """
    Test BoolBaseParser with a valid boolean input.
    """
    parser = BoolParser()
    result = parser.parse("true")
    assert result is True


def test_bool_parser_with_invalid_input():
    """
    Test BoolBaseParser with an invalid (non-boolean) input.
    """
    parser = BoolParser()
    with pytest.raises(ValueError):
        parser.parse("abc")


def test_triple_backtick_parser_with_valid_input():
    """
    Test TripleBacktickParser with a valid triple backtick input.
    """
    parser = TripleBacktickParser()
    result = parser.parse("```\nHello, World!\n```")
    assert result == "Hello, World!"


def test_triple_backtick_parser_with_invalid_input():
    """
    Test TripleBacktickParser with an invalid (non-triple backtick) input.
    """
    parser = TripleBacktickParser()
    with pytest.raises(ValueError):
        parser.parse("Hello, World!")


def test_toml_parser_with_valid_input():
    """
    Test TOMLParser with a valid TOML input within triple backticks.
    """
    parser = TOMLParser()
    result = parser.parse('```\nsome_key = "value"\n```')
    assert result == {"some_key": "value"}


def test_toml_parser_with_invalid_input():
    """
    Test TOMLParser with an invalid TOML input within triple backticks.
    """
    parser = TOMLParser()
    with pytest.raises(tomllib.TOMLDecodeError):
        parser.parse("```\nsome_key = value\n```")


def test_pydantic_parser_with_valid_input():
    """
    Test PydanticParser with valid JSON input within triple backticks.
    """
    parser = PydanticParser(Person)
    input_json = """```json
    {
        "name": "John Doe",
        "age": 30,
        "email": "john@example.com"
    }
    ```"""
    result = parser.parse(input_json)

    assert isinstance(result, Person)
    assert result.name == "John Doe"
    assert result.age == 30
    assert result.email == "john@example.com"


def test_pydantic_parser_without_backticks():
    """
    Test PydanticParser with raw JSON input (no triple backticks).
    """
    parser = PydanticParser(Person, triple_backticks=False)
    input_json = """
    {
        "name": "Jane Smith",
        "age": 25,
        "email": "jane@example.com"
    }
    """
    result = parser.parse(input_json)

    assert isinstance(result, Person)
    assert result.name == "Jane Smith"
    assert result.age == 25
    assert result.email == "jane@example.com"


def test_pydantic_parser_with_prefill():
    """
    Test PydanticParser with prefill string.
    """
    parser = PydanticParser(Person, prefill='```json\n{"name": "Alice", ')
    input_json = """
    "age": 28,
    "email": "alice@example.com"
}
    ```"""
    result = parser.parse(input_json)

    assert isinstance(result, Person)
    assert result.name == "Alice"
    assert result.age == 28
    assert result.email == "alice@example.com"


def test_pydantic_parser_with_optional_fields():
    """
    Test PydanticParser with a model containing optional fields.
    """
    parser = PydanticParser(Address)
    input_json = """```json
    {
        "street": "123 Main St",
        "city": "Anytown"
    }
    ```"""
    result = parser.parse(input_json)

    assert isinstance(result, Address)
    assert result.street == "123 Main St"
    assert result.city == "Anytown"
    assert result.zipcode is None


def test_pydantic_parser_with_invalid_json():
    """
    Test PydanticParser with invalid JSON input.
    """
    parser = PydanticParser(Person)
    input_json = """```json
    {
        "name": "John Doe"
        "age": 30,
        "email": "john@example.com"
    }
    ```"""

    with pytest.raises(ValueError, match="PydanticParser failed to parse JSON"):
        parser.parse(input_json)


def test_pydantic_parser_with_validation_error():
    """
    Test PydanticParser with JSON that fails pydantic validation.
    """
    parser = PydanticParser(Person)
    input_json = """```json
    {
        "name": "John Doe",
        "age": "thirty",
        "email": "john@example.com"
    }
    ```"""

    with pytest.raises(ValueError, match="PydanticParser validation failed"):
        parser.parse(input_json)


def test_pydantic_parser_missing_required_fields():
    """
    Test PydanticParser with JSON missing required fields.
    """
    parser = PydanticParser(Person)
    input_json = """```json
    {
        "name": "John Doe"
    }
    ```"""

    with pytest.raises(ValueError, match="PydanticParser validation failed"):
        parser.parse(input_json)


def test_pydantic_parser_no_backticks_error():
    """
    Test PydanticParser when expecting backticks but none are found.
    """
    parser = PydanticParser(Person, triple_backticks=True)
    input_json = """
    {
        "name": "John Doe",
        "age": 30,
        "email": "john@example.com"
    }
    """

    with pytest.raises(ValueError, match="PydanticParser found no triple backticks"):
        parser.parse(input_json)


def test_pydantic_parser_with_json_prefix():
    """
    Test PydanticParser with 'json' prefix in backticks.
    """
    parser = PydanticParser(Person)
    input_json = """```json
    {
        "name": "Bob Wilson",
        "age": 35,
        "email": "bob@example.com"
    }
    ```"""
    result = parser.parse(input_json)

    assert isinstance(result, Person)
    assert result.name == "Bob Wilson"
    assert result.age == 35
    assert result.email == "bob@example.com"
