import sys
from pprint import pformat

from demo.demo_policy.formatter import PolicyFormatter
from models import Policy
from fence.utils.logger import setup_logging
from fence.utils.optim import retry, parallelize
from fence import Link, Message, Messages, MessagesTemplate, TOMLParser, ClaudeHaiku, ClaudeSonnet
from fence.parsers import TripleBacktickParser
from prompts import SYSTEM_PROMPT_REFLECT, USER_PROMPT_REFLECT, ASSISTANT_PROMPT_REFLECT, USER_PROMPT_REVISE, SYSTEM_PROMPT_REVISE, ASSISTANT_PROMPT_REVISE
logger = setup_logging(__name__, log_level="INFO", serious_mode=False)

def handler(event: dict, context: any) -> dict:
    """
    Handler for the demo_cook lambda.
    """

    logger.info("👋 Let's rock!")

    # Set model
    claude_model = ClaudeSonnet(source='test_policies', region='us-east-1')

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
        parser=TOMLParser(prefill="```toml\nevaluation =")
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
        if result['evaluation'] == "<NON_COMPLIANT>":
            instructions.extend(result.get("instructions", []))
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
        parser=TripleBacktickParser()
    )

    # Run link for revised text
    revised_text = link.run(text=input_text, policies=full_policies, instructions=formatted_instructions)["revised_output"]

    # Build response
    return {"statusCode": 200, "body": {"revised_text": revised_text, "instructions": instructions}}


if __name__ == "__main__":

    # Set some example snippets
    snippet = """
        Ні,
        I should send over a demo first. Our MVP is ready to ship. I hope it doesn't fall on deaf
        ears.
        2 members of your Team should test, at minimum.
        It being used by revenue team members, is key. After all, Showpad is all about revenue enablement.
        Wdyt?
        Bram
        """

    snippet2 = """"Hi team,
Thanks for joining the meeting earlier today. 1 thing to consider is the time difference. Appreciate Everyone making it to the meeting even in GTM time zone.
FYI, the meeting has been recorded by Michael. So if you need the recording, do not hesitate to click on the red colour icon. I highly recommend checking if something is wrong.
Best,
Mehedi"""

    # Set some example policies
    policies = [
        Policy(
            policyType="text",
            value="Don't randomly capitalize words in the middle of sentences, unless they are supposed to be capitalized.",
            examples=[
                {"type": "positive", "value": "I like to eat apples."},
                {"type": "negative", "value": "I like to Eat apples."},
            ],
        ),
        Policy(
            policyType="text",
            value="Don't use idioms relating to disability, gender, ethnicity or religion.",
            examples=[
                {"type": "positive", "value": "I'm on cloud nine."},
                {"type": "negative", "value": "I'm on cloud seven."},
            ],
        ),
        Policy(
            policyType="text",
            value="Use the pronoun 'we' instead of 'I'.",
            examples=[
                {"type": "positive", "value": "We like to eat apples."},
                {"type": "negative", "value": "I like to eat apples."},
            ],
        ),
        Policy(
            policyType="text",
            value="Ensure proper spelling and grammar.",
        ),
        Policy(
            policyType="text",
            value="Spell out a number when it begins a sentence. Otherwise, use the numeral.",
        ),
        Policy(
            policyType="text",
            value="Spell out abbreviations the first time you mention it.",
        ),
    ]

    # Build event
    event = {"input": snippet, "policies": policies}

    # Run handler
    result = handler(event, None)

    # Print result
    logger.info(f"Result:\n{pformat(result)}")
    logger.critical(f"Revised text:\n{result['body']['revised_text']}")