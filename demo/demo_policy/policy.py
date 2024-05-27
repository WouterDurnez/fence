import sys
from pprint import pformat

from demo.demo_policy.formatter import PolicyFormatter
from presets import presets
from examples import examples
from models import Policy
from fence.utils.logger import setup_logging
from fence.utils.optim import retry, parallelize
from fence import (
    Link,
    Message,
    Messages,
    MessagesTemplate,
    TOMLParser,
    ClaudeHaiku,
    ClaudeSonnet,
)
from fence.parsers import TripleBacktickParser
from prompts import (
    SYSTEM_PROMPT_REFLECT,
    USER_PROMPT_REFLECT,
    ASSISTANT_PROMPT_REFLECT,
    USER_PROMPT_REVISE,
    SYSTEM_PROMPT_REVISE,
    ASSISTANT_PROMPT_REVISE,
)

logger = setup_logging(__name__, log_level="INFO", serious_mode=False)


def handler(event: dict, context: any) -> dict:
    """
    Handler for the demo_cook lambda.
    """

    logger.info("👋 Let's rock!")

    # Set model
    claude_model = ClaudeSonnet(
        source="test_policies", region="us-east-1", temperature=0
    )

    # Parse event
    input_text = event.get("input", "")
    policies = event.get("policies", [])

    logger.info(f"Input text: {input_text}")
    logger.info(f"Policies: {policies}")

    # Format policies
    formatter = PolicyFormatter()
    formatted_policies = []
    for policy in policies:
        formatted_policies.append(formatter.format_single(policy))
    full_policies = "\n".join(formatted_policies)

    # Create MessageTemplate
    user_message = Message(content=USER_PROMPT_REFLECT, role="user")
    assistant_message = Message(content=ASSISTANT_PROMPT_REFLECT, role="assistant")
    template = MessagesTemplate(
        source=Messages(
            system=SYSTEM_PROMPT_REFLECT,
            messages=[user_message, assistant_message],
        )
    )

    # Create link
    link = Link(
        llm=claude_model,
        name="reflect_link",
        template=template,
        output_key="reflect_output",
        parser=TOMLParser(prefill="```toml\nevaluation ="),
    )

    # Run link for policies in parallel
    @parallelize(max_workers=8)
    @retry(max_retries=3)
    def run_link(policy: str):
        result = link.run(text=input_text, policy=policy)["reflect_output"]
        return result

    results = run_link(formatted_policies)

    # Merge instructions, which are lists of strings
    instructions = []
    for result in results:
        if result["evaluation"] == "<NON_COMPLIANT>":
            instructions.extend(
                result.get("instructions", [])
                if type(result.get("instructions")) == list
                else [result.get("instructions")]
            )
    formatted_instructions = "\n".join(instructions)

    # Create MessageTemplate
    user_message = Message(content=USER_PROMPT_REVISE, role="user")
    assistant_message = Message(content=ASSISTANT_PROMPT_REVISE, role="assistant")
    template = MessagesTemplate(
        source=Messages(
            system=SYSTEM_PROMPT_REVISE,
            messages=[user_message, assistant_message],
        )
    )

    # Create link
    link = Link(
        llm=claude_model,
        name="revise_link",
        template=template,
        output_key="revised_output",
        parser=TripleBacktickParser(),
    )

    # Run link for revised text
    revised_text = link.run(
        text=input_text, policies=full_policies, instructions=formatted_instructions
    )["revised_output"]

    # Build response
    return {
        "statusCode": 200,
        "body": {"revised_text": revised_text, "instructions": instructions},
    }


if __name__ == "__main__":

    # Set the policies
    policies = [Policy(**preset) for preset in presets]

    def run_example(example):

        # Build event
        event = {"input": example, "policies": policies}

        # Run handler
        result = handler(event, None)

        # Return
        return result['body']['revised_text']

    # Build event
    outputs = [run_example(example) for example in examples]
