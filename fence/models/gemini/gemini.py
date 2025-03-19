"""
Google Gemini models
"""

import json
import logging
import os

import requests

from fence.models.base import LLM, MessagesMixin, get_log_callback
from fence.templates.messages import Message, Messages, MessagesTemplate

logger = logging.getLogger(__name__)


class GeminiBase(LLM):
    """Base class for Gemini models"""

    inference_type = "google-gemini"

    def __init__(
        self,
        source: str,
        metric_prefix: str | None = None,
        extra_tags: dict | None = None,
        endpoint: str | None = None,
        api_key: str | None = None,
        **kwargs,
    ):
        """
        Initialize a Gemini model

        :param str source: An indicator of where (e.g., which feature) the model is operating from.
        :param str|None metric_prefix: Prefix for the metric names
        :param dict|None extra_tags: Additional tags to add to the logging tags
        :param str|None api_key: Google Gemini API key
        :param **kwargs: Additional keyword arguments
        """

        super().__init__(
            source=source, metric_prefix=metric_prefix, extra_tags=extra_tags
        )
        # Model parameters
        self.model_kwargs = {
            "temperature": kwargs.get("temperature", 1),
            "max_tokens": kwargs.get("max_tokens", None),
        }

        # Find API key
        self.api_key = api_key or os.environ.get("GOOGLE_GEMINI_API_KEY", None)
        if not self.api_key:
            raise ValueError(
                "Google Gemini API key must be provided, either as an argument or in the environment variable `GOOGLE_GEMINI_API_KEY`"
            )

        self.endpoint = (
            endpoint
            or "https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent"
        )

    def invoke(self, prompt: str | Messages, **kwargs) -> str:
        """
        Call the model with the given prompt
        :param prompt: text to feed the model
        :return: response
        """

        self._check_if_prompt_is_valid(prompt=prompt)

        # Call the model
        response = self._invoke(prompt=prompt)

        # Get completion
        completion = response["candidates"][0]["content"]["parts"][0]["text"]

        # Get input and output tokens
        input_token_count = response["usageMetadata"]["promptTokenCount"]
        output_token_count = response["usageMetadata"]["candidatesTokenCount"]

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

        # Log all metrics if a log callback is registered
        if log_callback := get_log_callback():
            prefix = ".".join(
                item for item in [self.metric_prefix, self.source] if item
            )

            log_args = [
                # Add metrics
                {
                    f"{prefix}.invocation": 1,
                    f"{prefix}.input_token_count": input_token_count,
                    f"{prefix}.output_token_count": output_token_count,
                    f"{prefix}.input_word_count": input_word_count,
                    f"{prefix}.output_word_count": output_word_count,
                },
                # Add tags
                self.logging_tags,
            ]

            # Log metrics
            logger.debug(f"Logging args: {log_args}")
            log_callback(*log_args)

        return completion

    def _invoke(self, prompt: str | Messages) -> dict:
        """
        Handle the API request to the service
        :param prompt: text to feed the model
        :return: response completion
        """

        if isinstance(prompt, Messages):
            request_body = prompt.model_dump_gemini()
        elif isinstance(prompt, str):
            request_body = {"contents": [{"parts": [{"text": prompt}]}]}
        else:
            raise ValueError(
                f"Prompt must be a string or a list of messages. Got {type(prompt)}"
            )

        logger.debug(f"Request body: {request_body}")

        try:

            # send request to Google
            headers = {
                "Content-Type": "application/json",
            }

            params = {
                "key": self.api_key,
            }

            response = requests.post(
                url=self.endpoint.format(model_id=self.model_id),
                headers=headers,
                params=params,
                data=json.dumps(request_body),
            )

            if response.status_code != 200:
                raise ValueError(
                    f"Error raise by Google Gemini service: {response.text}"
                )

            # Parse the response
            response = response.json()

            return response

        except Exception as e:
            raise ValueError(f"Error raised by Google Gemini service: {e}")


class Gemini(GeminiBase, MessagesMixin):
    """Generic Gemini interface"""

    def __init__(self, model_id: str, source: str | None = None, **kwargs):
        """
        Initialize the Ollama model
        :param str model_id: The model ID as defined in the Ollama service
        :param str source: An indicator of where (e.g., which feature) the model is operating from.
        :param bool auto_pull: Whether to automatically pull the model if it is not found
        :param **kwargs: Additional keyword arguments
        """

        super().__init__(source=source, **kwargs)

        self.model_id = model_id

        # Split up model id using hyphens and capitalize each word
        self.model_name = " ".join([word.capitalize() for word in model_id.split("-")])


class GeminiFlash2_0(Gemini):
    """Gemini 2.0 Flash model"""

    def __init__(self, source: str | None = None, **kwargs):
        super().__init__(model_id="gemini-2.0-flash", source=source, **kwargs)


class GeminiFlash1_5(Gemini):
    """Gemini 1.5 Flash model"""

    def __init__(self, source: str | None = None, **kwargs):
        super().__init__(model_id="gemini-1.5-flash", source=source, **kwargs)


class Gemini1_5_Pro(Gemini):
    """Gemini 1.5 Pro model"""

    def __init__(self, source: str | None = None, **kwargs):
        super().__init__(model_id="gemini-1.5-pro", source=source, **kwargs)


if __name__ == "__main__":

    gemini = Gemini1_5_Pro(source="test")
    # Test prompt
    prompt = "Hello, how are you today?"

    # Call the model
    response = gemini.invoke(prompt=prompt)

    print(response)

    # You can also use the MessagesTemplates for more complex prompts
    messages = Messages(
        system="Respond in a {tone} tone",
        messages=[
            Message(role="user", content="Why is the sky {color}?"),
            # Equivalent to Message(role='user', content=Content(type='text', text='Why is the sky blue?'))
            # But Content can also be an image, etc.
        ],
    )
    messages_template = MessagesTemplate(source=messages)

    response = gemini.invoke(
        prompt=messages_template.render(tone="sarcastic", color="blue")
    )

    print(response)
