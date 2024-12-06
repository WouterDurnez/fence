import tomllib

import pytest

from fence.parsers import BoolParser, IntParser, TOMLParser, TripleBacktickParser


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
