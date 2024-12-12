"""
Some convenient shortcuts for common operations.
"""

from fence.links import Link
from fence.models.base import LLM
from fence.parsers import TOMLParser, TripleBacktickParser
from fence.templates.messages import MessagesTemplate
from fence.templates.models import Message, Messages


def create_toml_link(
    model: LLM,
    user_message: str,
    system_message: str | None = None,
    assistant_message: str | None = "```toml",
    name: str = "toml_link",
    **link_kwargs,
) -> Link:
    """
    Create a link object expected to return a TOML-formatted message.

    :param LLM model: The model object.
    :param str | None system_message: A system message.
    :param str user_message: The user message.
    :param str assistant_message: The assistant_message message.
    :param str name: The name of the link.
    :return: A link object containing the messages.
    """

    # Prefill is the assistant_message message, if any
    prefill = assistant_message

    # Create a message object
    user_message = Message(role="user", content=user_message)
    messages = Messages(messages=[user_message])

    # Add the assistant_message message, if any
    if assistant_message:
        assistant_message = Message(role="assistant", content=assistant_message)
        messages.messages.append(assistant_message)

    # Add the system_message message, if any
    if system_message:
        messages.system = system_message

    # Create a messages template
    template = MessagesTemplate(source=messages)

    # Initialize the link kwargs
    link_kwargs.update(
        {"model": model, "template": template, "name": name},
        parser=TOMLParser(prefill=prefill),
    )

    # Create a link
    link = Link(**link_kwargs)

    return link


def create_triple_backtick_link(
    model: LLM,
    user_message: str,
    system_message: str | None = None,
    assistant_message: str | None = "```",
    name: str = "triple_backtick_link",
    **link_kwargs,
) -> Link:
    """
    Create a link object expected to return a TOML-formatted message.

    :param LLM model: The model object.
    :param str | None system_message: A system message.
    :param str user_message: The user message.
    :param str assistant_message: The assistant_message message.
    :param str name: The name of the link.
    :return: A link object containing the messages.
    """

    # Prefill is the assistant_message message, if any
    prefill = assistant_message

    # Create a message object
    user_message = Message(role="user", content=user_message)
    messages = Messages(messages=[user_message])

    # Add the assistant_message message, if any
    if assistant_message:
        assistant_message = Message(role="assistant", content=assistant_message)
        messages.messages.append(assistant_message)

    # Add the system_message message, if any
    if system_message:
        messages.system = system_message

    # Create a messages template
    template = MessagesTemplate(source=messages)

    # Initialize the link kwargs
    link_kwargs.update(
        {"model": model, "template": template, "name": name},
        parser=TripleBacktickParser(prefill=prefill),
    )

    # Create a link
    link = Link(**link_kwargs)

    return link


def create_string_link(
    model: LLM,
    user_message: str,
    assistant_message: str | None = None,
    system_message: str | None = None,
    name: str = "string_link",
    **link_kwargs,
) -> Link:
    """
    Create a link object expected to return a string message.

    :param LLM model: The model object.
    :param str | None system_message: A system message.
    :param str user_message: The user message.
    :param str assistant_message: The assistant_message message.
    :param str name: The name of the link.
    :return: A link object containing the messages.
    """

    # Create a message object
    user_message = Message(role="user", content=user_message)
    messages = Messages(messages=[user_message])

    # Add the assistant_message message, if any
    if assistant_message:
        assistant_message = Message(role="assistant", content=assistant_message)
        messages.messages.append(assistant_message)

    # Add the system_message message, if any
    if system_message:
        messages.system = system_message

    # Create a messages template
    template = MessagesTemplate(source=messages)

    # Initialize the link kwargs
    link_kwargs.update({"model": model, "template": template, "name": name})

    # Create a link
    link = Link(**link_kwargs)

    return link


if __name__ == "__main__":
    from fence.models.openai import GPT4omini
    from fence.utils.logger import setup_logging

    logger = setup_logging(log_level="debug", are_you_serious=False)

    link = create_toml_link(
        model=GPT4omini(),
        user_message="Create an ingredient list for {recipe_name}",
        assistant_message="```toml",
        system_message="""You create ingredient lists in a TOML format. Here's an example:
[[ingredients]]
name = "Flour"
quantity = "2 cups"

[[ingredients]]
name = "Sugar"
quantity = "1 cup"

[[ingredients]]
name = "Eggs"
quantity = "2"

[[ingredients]]
name = "Milk"
quantity = "1.5 cups"
""",
    )

    result = link.run(recipe_name="chocolate cake")["state"]
    print(result)
