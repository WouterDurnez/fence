import inspect
import re
from typing import Callable


class DocstringParser:
    """Parser for extracting parameter descriptions from function docstrings.

    Supports multiple docstring formats:
    - ReST/Sphinx style: :param name: description
    - ReST with types: :param type name: description
    - Google style: Args: name: description
    - NumPy style: Parameters: name : type, description
    """

    def parse(self, method: Callable) -> dict[str, str | None]:
        """Parse the docstring of a method to get parameter descriptions.

        :param method: The method whose docstring is to be parsed.
        :return: A dict of param name -> description, None if no description found.
        """
        try:
            params = inspect.signature(method).parameters
        except (ValueError, TypeError):
            return {}

        if not params:
            return {}

        # Initialize with None values to match return type
        param_descriptions = {param: None for param in params}

        # Return early if no docstring
        if not hasattr(method, "__doc__") or not method.__doc__:
            return param_descriptions

        method_docstring = method.__doc__.strip()

        # Try different parsing strategies
        parsed = (
            self._parse_rest_style(method_docstring, param_descriptions)
            or self._parse_google_style(method_docstring, param_descriptions)
            or self._parse_numpy_style(method_docstring, param_descriptions)
        )

        return parsed if parsed else param_descriptions

    def _parse_rest_style(
        self, docstring: str, param_descriptions: dict[str, str | None]
    ) -> dict[str, str | None] | None:
        """Parse ReST/Sphinx style docstrings."""
        lines = docstring.split("\n")
        found_any = False
        current_param = None
        current_description_lines = []

        for line in lines:
            stripped = line.strip()

            # Check if this line starts a new parameter
            rest_match = re.match(r":param\s+(?:\w+\s+)?(\w+):\s*(.*)", stripped)
            if rest_match:
                # Save previous parameter if exists
                if current_param and current_param in param_descriptions:
                    param_descriptions[current_param] = (
                        " ".join(current_description_lines).strip() or None
                    )
                    found_any = True

                # Start new parameter
                current_param = rest_match.group(1)
                description = rest_match.group(2).strip()
                current_description_lines = [description] if description else []

            elif current_param and stripped and not stripped.startswith(":"):
                # Continuation of current parameter description
                current_description_lines.append(stripped)
            elif stripped.startswith(":") and current_param:
                # New docstring field, save current param
                if current_param in param_descriptions:
                    param_descriptions[current_param] = (
                        " ".join(current_description_lines).strip() or None
                    )
                    found_any = True
                current_param = None
                current_description_lines = []

        # Handle last parameter
        if current_param and current_param in param_descriptions:
            param_descriptions[current_param] = (
                " ".join(current_description_lines).strip() or None
            )
            found_any = True

        return param_descriptions if found_any else None

    def _parse_google_style(
        self, docstring: str, param_descriptions: dict[str, str | None]
    ) -> dict[str, str | None] | None:
        """Parse Google style docstrings."""
        lines = docstring.split("\n")
        in_args_section = False
        found_any = False
        current_param = None
        current_description_lines = []

        for line in lines:
            stripped = line.strip()

            # Check for Args section
            if stripped.lower() in ["args:", "arguments:", "parameters:"]:
                in_args_section = True
                continue
            elif stripped.endswith(":") and not stripped.lower().startswith("arg"):
                # Different section started
                in_args_section = False
                if current_param and current_param in param_descriptions:
                    param_descriptions[current_param] = (
                        " ".join(current_description_lines).strip() or None
                    )
                    found_any = True
                current_param = None
                current_description_lines = []
                continue

            if in_args_section and stripped:
                # Check if this is a parameter line
                google_match = re.match(r"(\w+)(?:\s*\([^)]+\))?\s*:\s*(.*)", stripped)
                if google_match:
                    # Save previous parameter
                    if current_param and current_param in param_descriptions:
                        param_descriptions[current_param] = (
                            " ".join(current_description_lines).strip() or None
                        )
                        found_any = True

                    # Start new parameter
                    current_param = google_match.group(1)
                    description = google_match.group(2).strip()
                    current_description_lines = [description] if description else []

                elif current_param and len(line) - len(line.lstrip()) > 0:
                    # Continuation line (indented)
                    current_description_lines.append(stripped)

        # Handle last parameter
        if current_param and current_param in param_descriptions:
            param_descriptions[current_param] = (
                " ".join(current_description_lines).strip() or None
            )
            found_any = True

        return param_descriptions if found_any else None

    def _parse_numpy_style(
        self, docstring: str, param_descriptions: dict[str, str | None]
    ) -> dict[str, str | None] | None:
        """Parse NumPy style docstrings."""
        lines = docstring.split("\n")
        in_params_section = False
        found_any = False
        current_param = None
        current_description_lines = []

        for line in lines:
            stripped = line.strip()

            # Check for Parameters section
            if stripped.lower() in ["parameters", "parameters:", "args", "args:"]:
                in_params_section = True
                continue
            elif stripped and not stripped.startswith("-") and stripped.endswith(":"):
                # Different section
                in_params_section = False
                if current_param and current_param in param_descriptions:
                    param_descriptions[current_param] = (
                        " ".join(current_description_lines).strip() or None
                    )
                    found_any = True
                current_param = None
                current_description_lines = []
                continue
            elif stripped.startswith("---"):
                # Section separator
                continue

            if in_params_section and stripped:
                # Check if this is a parameter line
                numpy_match = re.match(r"(\w+)\s*:\s*(?:[^,]+,?)?\s*(.*)", stripped)
                if numpy_match:
                    # Save previous parameter
                    if current_param and current_param in param_descriptions:
                        param_descriptions[current_param] = (
                            " ".join(current_description_lines).strip() or None
                        )
                        found_any = True

                    # Start new parameter
                    current_param = numpy_match.group(1)
                    description = numpy_match.group(2).strip()
                    current_description_lines = [description] if description else []

                elif current_param and (
                    line.startswith("    ") or line.startswith("\t")
                ):
                    # Continuation line (indented)
                    current_description_lines.append(stripped)

        # Handle last parameter
        if current_param and current_param in param_descriptions:
            param_descriptions[current_param] = (
                " ".join(current_description_lines).strip() or None
            )
            found_any = True

        return param_descriptions if found_any else None


# Convenience function for backward compatibility
def docstring_parser(method: Callable) -> dict[str, str | None]:
    """Parse the docstring of a method to get parameter descriptions.

    :param method: The method whose docstring is to be parsed.
    :return: A dict of param name -> description, None if no description found.
    """
    parser = DocstringParser()
    return parser.parse(method)
