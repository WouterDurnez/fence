"""
Mistral models
"""

import json
import os

import requests

import logging

from fence.models.base import LLM, MessagesMixin
from fence.templates.messages import Message, Messages, MessagesTemplate

logger = logging.getLogger(__name__)


class MistralBase(LLM, MessagesMixin):
    """Base class for Mistral models"""

    model_id = None
    model_name = None
    inference_type = "mistral"

    def __init__(
        self,
        source: str | None = None,
        metric_prefix: str | None = None,
        extra_tags: dict | None = None,
        api_key: str | None = None,
        **kwargs,
    ):
        """
        Initialize a Mistral model

        :param str source: An indicator of where (e.g., which feature) the model is operating from.
        :param str|None metric_prefix: Prefix for the metric names
        :param dict|None extra_tags: Additional tags to add to the logging tags
        :param str|None api_key: Mistral
        :param **kwargs: Additional keyword arguments
        """

        super().__init__(
            source=source, metric_prefix=metric_prefix, extra_tags=extra_tags
        )
        
        # Find API key
        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY", None)
        if not self.api_key:
            raise ValueError(
                "Mistral API key must be provided, either as an argument or in the environment variable `MISTRAL_API_KEY`"
            )
        
        self.url = "https://api.mistral.ai"
        self.chat_endpoint = self.url + "/v1/chat/completions"

        # LLM parameters
        self.model_kwargs = {
            "temperature": kwargs.get("temperature", None),
            "max_tokens": kwargs.get("max_tokens", None),
            "top_p": kwargs.get("top_p", 1),
            "stop": kwargs.get("stop", None),
        }

    def invoke(self, prompt: str | Messages, **kwargs) -> str:
        """
        Invoke the model

        :param str|Messages prompt: The prompt to send to the model
        :param **kwargs: Additional keyword arguments
        :return: The model's response
        :rtype: str
        """
        
        # Prompt should not be empty
        self._check_if_prompt_is_valid(prompt=prompt)

        # Call the model
        response = self._invoke(prompt=prompt)

        # Get input and output tokens
        input_token_count = response["usage"]["prompt_tokens"]
        output_token_count = response["usage"]["completion_tokens"]

        # Get response completion
        completion = response["choices"][0]["message"]["content"]

        # Calculate word count
        if isinstance(prompt, str):
            input_word_count = len(prompt.split())
        elif isinstance(prompt, Messages):
            input_word_count = self.count_words_in_messages(prompt)
        else:
            raise ValueError(
                f"Prompt must be a string or a list of messages. Got {type(prompt)}"
            )
        output_word_count = len(completion.split())

        metrics = {
            "input_token_count": input_token_count,
            "output_token_count": output_token_count,
            "input_word_count": input_word_count,
            "output_word_count": output_word_count,
        }

        # Log all metrics if a log callback is registered
        self._log_callback(**metrics)

        # Calculate token metrics for the response and send them to Datadog
        return completion


    def _invoke(self, prompt: str | Messages, **kwargs) -> str:
        """
        Handle the API request to the service
        :param prompt: text to feed the model
        :return: response completion
        """

        # Format prompt, using OpenAIs formatting.
        # However, if we receive a string, we will format it as a single user message for ease of use.
        if isinstance(prompt, Messages):

            # Format messages
            messages = prompt.model_dump_mistral()

        elif isinstance(prompt, str):
            messages = [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
        else:
            raise ValueError(
                f"Prompt must be a string or a list of messages. Got {type(prompt)}"
            )

        model_kwargs = {k: v for k, v in self.model_kwargs.items() if v is not None}

        # Build request body
        request_body = {
            **model_kwargs,
            "messages": messages,
            "model": self.model_id,
        }

        logger.debug(f"Request body: {request_body}")

        # Send request
        try:

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            # Send request to Mistral
            response = requests.post(
                url=self.chat_endpoint, headers=headers, data=json.dumps(request_body)
            )

            # Check status code
            if response.status_code != 200:
                raise ValueError(f"Error raised by Mistral service: {response.text}")
            
            # Parse response
            response = response.json()

            return response
        
        except Exception as e:
            raise ValueError(f"Error raised by Mistral service: {e}")
        
class Mistral(MistralBase):
    """
    Mistral model
    """

    def __init__(self, model_id: str, **kwargs):
        """
        Initialize a Mistral model

        :param str model_id: The Mistral model ID
        :param **kwargs: Additional keyword arguments
        """

        super().__init__(**kwargs)
        self.model_id = model_id

        self.model_name = self.model_id.capitalize()


if __name__ == "__main__":

    model = Mistral(model_id="mistral-large-latest")

    prompt = "What is the meaning of life?"
    response = model(prompt)
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

    response = model.invoke(
        prompt=messages_template.render(tone="sarcastic", color="blue")
    )

    print(response)