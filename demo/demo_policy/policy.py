import sys
from datetime import datetime
from pprint import pformat

from demo.demo_policy.formatter import PolicyFormatter
from demo.demo_policy.utils import format_feedback
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
    SYSTEM_PROMPT_REVISE_FEEDBACK,
    ASSISTANT_PROMPT_REVISE,
)

logger = setup_logging(__name__, log_level="INFO", serious_mode=False)


def handler(event: dict, context: any) -> dict:
    """
    Handler for the demo_cook lambda.
    """

    logger.info("👋 Let's rock!")

    # Set model
    claude_model = ClaudeHaiku(
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
        formatted_policies.append(formatter.format_single(policy, examples=False))
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
    instructions, suggestions, feedback = [], [], []

    # Extract instructions and suggestions
    for policy, result in zip(policies, results):

        # Check if the policy is non-compliant, we don't need instructions otherwise
        if result["evaluation"] == "<NON_COMPLIANT>":
            policies.append(policy)
            suggestion = result.get("suggested_text", "")
            instruction = result.get("instructions", []) if type(result.get("instructions")) == list  else [result.get("instructions")]
            instructions.extend(instruction)
            suggestions.append(suggestion)

            # Create feedback package
            feedback.append(
                {
                    "policy": policy.value,
                    "instructions": instruction,
                    "suggestions": suggestion,
                }
            )
            #logger.critical(pformat(feedback))

    formatted_instructions = "[INSTRUCTIONS]\n" + "\n".join(instructions)
    formatted_suggestions = "[SUGGESTIONS]\n" + "\n".join(suggestions)
    formatted_feedback = format_feedback(feedback=feedback)

    # Create MessageTemplate
    user_message = Message(content=USER_PROMPT_REVISE, role="user")
    assistant_message = Message(content=ASSISTANT_PROMPT_REVISE, role="assistant")
    template = MessagesTemplate(
        source=Messages(
            system=SYSTEM_PROMPT_REVISE_FEEDBACK,
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
    # revised_text = link.run(
    #     text=input_text, policies=full_policies, instructions=formatted_instructions
    # )["revised_output"]
    revised_text = link.run(
        text=input_text, feedback=formatted_feedback
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

    BATCH = True

    if BATCH:
        # Build event
        outputs = parallelize(max_workers=8)(run_example)(examples)

        # Save outputs to file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        with open(f"outputs_{timestamp}.txt", "w") as f:
            for index, (input, output) in enumerate(zip(examples,outputs)):
                f.write(f"[INPUT {index}]\n")
                f.write(input)
                f.write("\n")
                f.write(f"[OUTPUT {index}]\n")
                f.write(output)
                f.write("\n\n")
    else:
        single_example = examples[0]
        revised_text = run_example(single_example)
        logger.warning(f"Example: {single_example}")
        logger.critical(f"Revised text: {revised_text}")
