
import logging
from models import Policy

logger = logging.getLogger(__name__)
class PolicyFormatter:
    """
    Given a policy object, which is a list of Policy objects, this class helps to format the policy in an LLM-friendly way.
    """

    def __init__(self, ):
        pass
    def format_single(self, policy: Policy, indent_examples=None) -> str:

        # Set indent_examples to ""
        if indent_examples is None:
            indent_examples = ""

        # Get the policy type and value
        policy_value = policy.value

        # Initialize the formatted policy
        formatted_policy = f"<policy>\n\n{policy_value}\n\n"

        # Check if the policy has examples
        if policy.examples:

            # Get positive and negative examples
            positive_examples = [
                example.value
                for example in policy.examples
                if example.type == "positive"
            ]
            negative_examples = [
                example.value
                for example in policy.examples
                if example.type == "negative"
            ]

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

    def format(self, policies: list[Policy]) -> str:

        # Initialize the formatted policy
        formatted_policy = ""

        # Format each policy
        for policy in policies:
            formatted_policy += self.format_single(policy) + "\n\n"

        return formatted_policy

if __name__ == '__main__':

    policy = [
        {
            "policyType": "text", "value": "Don't use contractions.",
            "examples": [
                {"type": "positive", "value": "Do not."},
                {"type": "negative", "value": "Don't."}
            ],
        },
    ]

    policy_formatter = PolicyFormatter(policy)
    formatted_policy = policy_formatter.format()
    print(formatted_policy)