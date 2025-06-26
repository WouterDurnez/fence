"""
Docstring parser tests
"""

from fence.utils.docstring_parser import DocstringParser, docstring_parser


class TestDocstringParser:
    """Test suite for the DocstringParser class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.parser = DocstringParser()

    def test_rest_style_basic(self):
        """Test basic ReST style docstring parsing."""

        def sample_function(name: str, age: int):
            """Sample function with ReST docstring.

            :param name: The person's name
            :param age: The person's age
            """
            pass

        result = self.parser.parse(sample_function)

        assert result == {"name": "The person's name", "age": "The person's age"}

    def test_rest_style_with_types(self):
        """Test ReST style with type annotations."""

        def sample_function(name: str, age: int):
            """Sample function with typed ReST docstring.

            :param str name: The person's name
            :param int age: The person's age
            """
            pass

        result = self.parser.parse(sample_function)

        assert result == {"name": "The person's name", "age": "The person's age"}

    def test_rest_style_multiline(self):
        """Test ReST style with multi-line descriptions."""

        def sample_function(name: str, config: dict):
            """Sample function with multi-line descriptions.

            :param name: The person's name as a string
            :param config: Configuration dictionary containing
                various settings and parameters for the function
                to operate correctly
            """
            pass

        result = self.parser.parse(sample_function)

        assert result == {
            "name": "The person's name as a string",
            "config": "Configuration dictionary containing various settings and parameters for the function to operate correctly",
        }

    def test_google_style_basic(self):
        """Test basic Google style docstring parsing."""

        def sample_function(name: str, age: int):
            """Sample function with Google docstring.

            Args:
                name: The person's name
                age: The person's age
            """
            pass

        result = self.parser.parse(sample_function)

        assert result == {"name": "The person's name", "age": "The person's age"}

    def test_google_style_with_types(self):
        """Test Google style with type annotations."""

        def sample_function(name: str, age: int):
            """Sample function with typed Google docstring.

            Args:
                name (str): The person's name
                age (int): The person's age
            """
            pass

        result = self.parser.parse(sample_function)

        assert result == {"name": "The person's name", "age": "The person's age"}

    def test_google_style_multiline(self):
        """Test Google style with multi-line descriptions."""

        def sample_function(name: str, config: dict):
            """Sample function with multi-line descriptions.

            Args:
                name: The person's name as a string
                config: Configuration dictionary containing
                    various settings and parameters for the function
                    to operate correctly
            """
            pass

        result = self.parser.parse(sample_function)

        assert result == {
            "name": "The person's name as a string",
            "config": "Configuration dictionary containing various settings and parameters for the function to operate correctly",
        }

    def test_numpy_style_basic(self):
        """Test basic NumPy style docstring parsing."""

        def sample_function(name: str, age: int):
            """Sample function with NumPy docstring.

            Parameters
            ----------
            name : str
                The person's name
            age : int
                The person's age
            """
            pass

        result = self.parser.parse(sample_function)

        assert result == {"name": "The person's name", "age": "The person's age"}

    def test_numpy_style_optional(self):
        """Test NumPy style with optional parameters."""

        def sample_function(name: str, age: int, active: bool = True):
            """Sample function with optional parameters.

            Parameters
            ----------
            name : str
                The person's name
            age : int
                The person's age
            active : bool, optional
                Whether the person is active
            """
            pass

        result = self.parser.parse(sample_function)

        assert result == {
            "name": "The person's name",
            "age": "The person's age",
            "active": "optional Whether the person is active",
        }

    def test_numpy_style_multiline(self):
        """Test NumPy style with multi-line descriptions."""

        def sample_function(name: str, config: dict):
            """Sample function with multi-line descriptions.

            Parameters
            ----------
            name : str
                The person's name as a string
            config : dict
                Configuration dictionary containing
                various settings and parameters for the function
                to operate correctly
            """
            pass

        result = self.parser.parse(sample_function)

        assert result == {
            "name": "The person's name as a string",
            "config": "Configuration dictionary containing various settings and parameters for the function to operate correctly",
        }

    def test_no_docstring(self):
        """Test function with no docstring."""

        def sample_function(name: str, age: int):
            pass

        result = self.parser.parse(sample_function)

        assert result == {"name": None, "age": None}

    def test_empty_docstring(self):
        """Test function with empty docstring."""

        def sample_function(name: str, age: int):
            """"""
            pass

        result = self.parser.parse(sample_function)

        assert result == {"name": None, "age": None}

    def test_docstring_without_params(self):
        """Test function with docstring but no parameter documentation."""

        def sample_function(name: str, age: int):
            """This function does something but doesn't document parameters."""
            pass

        result = self.parser.parse(sample_function)

        assert result == {"name": None, "age": None}

    def test_no_parameters(self):
        """Test function with no parameters."""

        def sample_function():
            """Function with no parameters.

            :return: Nothing
            """
            pass

        result = self.parser.parse(sample_function)

        assert result == {}

    def test_partial_documentation(self):
        """Test function with partial parameter documentation."""

        def sample_function(name: str, age: int, active: bool):
            """Function with partial documentation.

            :param name: The person's name
            :param active: Whether active
            """
            pass

        result = self.parser.parse(sample_function)

        assert result == {
            "name": "The person's name",
            "age": None,
            "active": "Whether active",
        }

    def test_mixed_format_fallback(self):
        """Test that parser falls back correctly when first format fails."""

        def sample_function(name: str, age: int):
            """Function with Google-style docstring.

            This should not be parsed as ReST style.

            Args:
                name: The person's name
                age: The person's age
            """
            pass

        result = self.parser.parse(sample_function)

        assert result == {"name": "The person's name", "age": "The person's age"}

    def test_malformed_signature(self):
        """Test handling of methods with problematic signatures."""

        # Create a mock object that will raise an exception during signature inspection
        class MockMethod:
            def __init__(self):
                self.__doc__ = ":param x: A parameter"

            def __call__(self):
                pass

        mock_method = MockMethod()
        # Remove the __code__ attribute to make inspect.signature fail
        if hasattr(mock_method, "__code__"):
            delattr(mock_method, "__code__")

        result = self.parser.parse(mock_method)

        assert result == {}

    def test_convenience_function(self):
        """Test the convenience function for backward compatibility."""

        def sample_function(name: str, age: int):
            """Sample function.

            :param name: The person's name
            :param age: The person's age
            """
            pass

        result = docstring_parser(sample_function)

        assert result == {"name": "The person's name", "age": "The person's age"}

    def test_special_parameter_names(self):
        """Test handling of parameters with special names."""

        def sample_function(self, _private: str, __dunder__: int):
            """Function with special parameter names.

            :param _private: A private parameter
            :param __dunder__: A dunder parameter
            """
            pass

        result = self.parser.parse(sample_function)

        assert result == {
            "self": None,
            "_private": "A private parameter",
            "__dunder__": "A dunder parameter",
        }

    def test_args_kwargs(self):
        """Test handling of *args and **kwargs."""

        def sample_function(name: str, *args, **kwargs):
            """Function with args and kwargs.

            :param name: The person's name
            :param args: Variable arguments
            :param kwargs: Keyword arguments
            """
            pass

        result = self.parser.parse(sample_function)

        assert result == {
            "name": "The person's name",
            "args": "Variable arguments",
            "kwargs": "Keyword arguments",
        }

    def test_case_insensitive_sections(self):
        """Test case-insensitive section headers."""

        def sample_function(name: str, age: int):
            """Function with case variations.

            ARGS:
                name: The person's name
                age: The person's age
            """
            pass

        # Our parser actually handles case-insensitive Google-style sections
        result = self.parser.parse(sample_function)

        assert result == {"name": "The person's name", "age": "The person's age"}
