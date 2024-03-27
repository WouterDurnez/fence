from fence import LLM, Link, PromptTemplate
from fence.demo.demo_policy.prompt_templates import (
    CONTEXTS,
    CORRECTION_TEMPLATE,
    GUIDELINES,
    POLICY_TEMPLATE,
    TONE_TEMPLATE,
    VERBOSITY_TEMPLATE,
)
from fence.src.llm.parsers import TOMLParser, TripleBacktickParser
from fence.src.utils.base import setup_logging
import re
from jinja2 import DebugUndefined, Template


logger = setup_logging(log_level="DEBUG")


class AIPlaceholder:
    """
    A class that provides methods for replacing paragraphs with placeholders, and vice versa. This allows us
    to process only specific sections of an input text, while leaving other sections unchanged.
    """

    start_marker = r"<!-- no-ai-start -->"
    stop_marker = r"<!-- no-ai-stop -->"

    def __init__(self, start_marker: str = None, stop_marker: str = None):
        """
        Initialize the AIPlaceholder object.
        :param start_marker: The start marker for the paragraphs that should NOT be processed by the AI.
        :param stop_marker: The stop marker for the paragraphs that should NOT be processed by the AI.
        """

        if start_marker:
            self.start_marker = start_marker
        if stop_marker:
            self.stop_marker = stop_marker

        # Store the mapping of placeholders to original paragraphs
        self.mapping = {}

    def replace_paragraphs_with_placeholders(self, text: str):
        """
        Replace paragraphs that are delimited by HTML comments with placeholders tags.
        :param text: The input text.
        :return: The text with the delimited paragraphs replaced by placeholders.
        """

        # Find all occurrences of the delimited paragraphs
        delimited_paragraphs = re.findall(
            f"{self.start_marker}(.*?){self.stop_marker}", text, re.DOTALL
        )
        logger.info(f"Found {len(delimited_paragraphs)} <no-ai> paragraphs")

        # If no delimited paragraphs are found, return the original text
        if not delimited_paragraphs:
            return text

        # Replace each delimited paragraph with a placeholder, and store the mapping
        for i, paragraph in enumerate(delimited_paragraphs):
            placeholder = f"<placeholder{i}>"
            self.mapping[placeholder] = paragraph
            text = text.replace(
                f"{self.start_marker}{paragraph}{self.stop_marker}", placeholder
            )

        return text

    def replace_placeholders_with_paragraphs(self, text: str):
        """
        Replace placeholders with the original paragraphs.
        :param text: The input text.
        :return: The text with the placeholders replaced by the original paragraphs.
        """

        # Replace each placeholder with the original paragraph
        for placeholder, paragraph in self.mapping.items():
            # Check if the placeholder is present in the text, and raise an error if it's not
            if placeholder not in text:
                logger.error(
                    f"Placeholder {placeholder} not found in the text: {text}."
                )
                raise ValueError(f"Placeholder {placeholder} not found in the text.")

            # Replace the placeholder with the original paragraph
            text = text.replace(
                placeholder, f"{self.start_marker}{paragraph}{self.stop_marker}"
            )

        # Find and remove any remaining placeholders (opening or closing tags)
        remaining_placeholders = re.findall(
            r"\s*(<placeholder\d+>|<\/placeholder\d+>)\s*", text
        )
        if remaining_placeholders:
            logger.warning(
                f"Found {len(remaining_placeholders)} remaining placeholders in the text. Removing them."
            )
            for placeholder in remaining_placeholders:
                text = text.replace(placeholder, "")

        # Cleanup step: strip and remove double spaces
        text = re.sub(r"\s+", " ", text.strip())

        return text

class PolicyFormatter:
    """
    Given a policy object, which is a list of Policy objects, this class helps to format the policy in an LLM-friendly way.

    The input format is as follows:

    [
        {
            "policyType": "text", "value": "Don't use contractions.",
            "examples": [
                {"type": "positive", "value": "Do not."},
                {"type": "negative", "value": "Don't."}
            ],
        },
    ]

    """

    def __init__(self, policy: list[dict]):
        self.policy = policy
        logger.debug(f"Formatting policy: {policy}")

    def _format_single_policy(self, policy: dict, indent_examples=None) -> str:

        # Set indent_examples to ""
        if indent_examples is None:
            indent_examples = ""

        # Get the policy type and value
        policy_value = policy.get("value", "")

        # Initialize the formatted policy
        formatted_policy = f"<policy>\n\n{policy_value}\n\n"

        # Check if the policy has examples
        if 'examples' in policy and policy['examples']:

            # Get positive and negative examples
            positive_examples = [example.get("value", "") for example in policy.get("examples", []) if
                                 example.get("type", "") == "positive"]
            negative_examples = [example.get("value", "") for example in policy.get("examples", []) if
                                 example.get("type", "") == "negative"]

            # Add the examples to the formatted policy
            if positive_examples:
                formatted_policy += f"{indent_examples}<positive_examples>\n"
                for example in positive_examples:
                    formatted_policy += f"{indent_examples}- {example}\n"
                formatted_policy += f"{indent_examples}</positive_examples>\n"

            if negative_examples:
                formatted_policy += f"{indent_examples}<negative_examples>\n"
                for example in negative_examples:
                    formatted_policy += f"{indent_examples}- {example}\n"
                formatted_policy += f"{indent_examples}</negative_examples>\n"

        formatted_policy += "\n</policy>"

        return formatted_policy

    def format(self) -> str:

        # Initialize the formatted policy
        formatted_policy = ""

        # Format each policy
        for policy in self.policy:
            formatted_policy += self._format_single_policy(policy) + "\n\n"

        return formatted_policy


def build_links(recipe: dict, llm: LLM, source: str=None) -> list[Link]:
    """
    Build and return a list of links based on the provided recipe and templates.
    :param recipe: recipe JSON object
    :param llm: LLM instance
    """
    links = []

    # Get the context for the source
    context = CONTEXTS.get(source, "")
    if not context:
        logger.warning(
            f"Source is not provided, or is not in the context dictionary: {source}"
        )

    # Create links for each key in the recipe
    for key in ["verbosity", "policy", "tone", "correction"]:
        # Check if the key is in the recipe and is not None
        if key in recipe and recipe[key] is not None:
            # Get modifier
            modifier = recipe[key] if key != "correction" else None

            # For verbosity and tone, we need to get the value
            if key in ["verbosity", "tone"]:
                modifier = modifier.value

            # If the key is `policy`, we need to format the policies and examples
            if key == "policy":

                modifier = PolicyFormatter(policy=modifier).format()

            # Prep template with context and guidelines
            template = Template(POLICY_TEMPLATE, undefined=DebugUndefined).render(
                context=context, guidelines=GUIDELINES, modifier=modifier
            )

            # Create link
            link = Link(
                name=f"{key}_link",
                template=PromptTemplate(
                    template=template, input_variables=["state", "recipe"]
                ),
                llm=llm,
                parser=TripleBacktickParser(),
                output_key=f"{key}_output",
            )
            links.append(link)

            logger.debug(f"Created link for {key} with template: {template}")

    return links

