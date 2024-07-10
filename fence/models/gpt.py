"""
OpenAI GPT models
"""

import logging

from openai import ChatCompletion, OpenAI

from fence.templates.messages import Message, Messages
from fence.utils.logger import setup_logging
from fence.models.base import LLM

logger = setup_logging(log_level="INFO", serious_mode=False)

# Suppress httpx logging message (nobody asked for you)
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)


class GPTBase(LLM):
    """Base class for GPT models"""

    model_name = None
    llm_name = None
    inference_type = "openai"

    def __init__(self, source: str, **kwargs):
        """
        Initialize a GPT model

        :param str source: An indicator of where (e.g., which feature) the model is operating from.
        :param **kwargs: Additional keyword arguments
        """

        self.source = source

        self.temperature = kwargs.get("temperature", 1)
        self.max_tokens = kwargs.get("max_tokens", None)

        self.model_kwargs = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        self.client = OpenAI()

    def invoke(self, prompt: str | Messages, **kwargs) -> str:
        """
        Call the model with the given prompt
        :param prompt: text to feed the model
        :return: response
        """

        # Call the model
        response = self._invoke(prompt=prompt)

        # Get input and output tokens
        input_token_count = response.usage.prompt_tokens
        output_token_count = response.usage.completion_tokens

        # Get response completion
        completion = response.choices[0].message.content

        # Get input and output word count
        if isinstance(prompt, str):
            input_word_count = len(prompt.split())
        elif isinstance(prompt, Messages):
            input_word_count = self.count_words_in_messages(prompt)
        else:
            raise ValueError(
                f"Prompt must be a string or a list of messages. Got {type(prompt)}"
            )
        output_word_count = len(completion.split())

        logger.debug(
            f"Token and word counts: {input_token_count=}, {output_token_count=}, {input_word_count=}, {output_word_count=} "
        )  # TODO: Remove once we can send this to Datadog

        # Calculate token metrics for the response and send them to Datadog
        return completion

    def _invoke(self, prompt: str | Messages) -> ChatCompletion:
        """
        Handle the API request to the service
        :param prompt: text to feed the model
        :return: response completion
        """

        # Format prompt, using OpenAIs formatting.
        # However, if we receive a string, we will format it as a single user message for ease of use.
        if isinstance(prompt, Messages):

            # Get messages
            messages = [
                message.model_dump(exclude_none=True) for message in prompt.messages
            ]
            system_message = prompt.system

            # Extract system message
            if system_message:
                messages.insert(0, {"role": "system", "content": "system_message"})
        elif isinstance(prompt, str):
            messages = [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]

        else:
            raise ValueError("Prompt must be a string or a list of messages")

        # Build request body
        request_body = {
            **self.model_kwargs,
            "messages": messages,
            "model": self.model_name,
        }

        logger.debug(f"Request body: {request_body}")

        # Send request
        try:
            response = self.client.chat.completions.create(**request_body)

            return response

        except Exception as e:
            raise ValueError(f"Error raised by OpenAI service: {e}")

    @staticmethod
    def count_words_in_messages(messages: Messages):
        """
        Count the number of words in a list of messages. Takes all roles into account. Type must be 'text'
        :param messages: list of messages
        :return: word count (int)
        """

        # Get user and assistant messages
        user_assistant_messages = messages.messages

        # Get system message
        system_messages = messages.system

        # Initialize word count
        word_count = 0

        # Loop through messages
        for message in user_assistant_messages:

            # Get content
            content = message.content

            # Content is either a string, or a list of content objects
            if isinstance(content, str):
                word_count += len(content.split())
            elif isinstance(content, list):
                for content_object in content:
                    if content_object.type == "text":
                        word_count += len(content_object.text.split())

        # Add system message to word count
        if system_messages:
            word_count += len(system_messages.split())

        return word_count


class GPT4o(GPTBase):
    """
    GPT-4o model
    """

    model_name = "gpt-4o"
    llm_name = "gpt-4o"

    def __init__(self, source: str, **kwargs):
        """
        Initialize a GPT-4o model
        :param source: An indicator of where (e.g., which feature) the model is operating from.
        :param **kwargs: Additional keyword arguments
        """

        super().__init__(source=source, **kwargs)


if __name__ == "__main__":

    # Initialize GPT model
    gpt = GPT4o(source="test")

    # Test prompt
    prompt = "Hello, how are you today?"

    # Call the model
    response = gpt(prompt)
    logger.info(f"GPT response: {response}")

    # Test with Messages
    messages = Messages(
        system="Respond in a very rude manner",
        messages=[Message(role="user", content="Hello, how are you today?")],
    )

    # Call the model
    response_rude = gpt(messages)
    logger.info(f"GPT response: {response_rude}")
