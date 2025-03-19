"""
Ollama models
"""

import logging

import requests
from requests import Response

from fence.models.base import LLM, MessagesMixin, get_log_callback
from fence.templates.models import Messages

logger = logging.getLogger(__name__)


class OllamaBase(LLM, MessagesMixin):
    """Base class for Ollama-powered models"""

    inference_type = "ollama"

    def __init__(
        self,
        source: str,
        metric_prefix: str | None = None,
        extra_tags: dict | None = None,
        endpoint: str | None = None,
        **kwargs,
    ):
        """
        Initialize a Claude model

        :param str source: An indicator of where (e.g., which feature) the model is operating from
        :param str|None metric_prefix: Prefix for the metric names
        :param dict|None extra_tags: Additional tags to add to the logging tags
        :param **kwargs: Additional keyword arguments
        """

        super().__init__(
            source=source, metric_prefix=metric_prefix, extra_tags=extra_tags
        )

        # Set default generate endpoint
        self.endpoint = endpoint or "http://localhost:11434/api"
        self.generate_endpoint = self.endpoint + "/generate"
        self.chat_endpoint = self.endpoint + "/chat"
        self.tag_endpoint = self.endpoint + "/tags"
        self.pull_endpoint = self.endpoint + "/pull"

        # Check if the endpoint is valid
        try:
            requests.get(self.tag_endpoint)
        except requests.exceptions.ConnectionError:
            raise ValueError(
                f"No Ollama service found at {self.endpoint}. Try installing it using `brew install ollama`."
            )

        # LLM parameters
        self.model_kwargs = {
            "temperature": kwargs.get("temperature", 0.01),
            "max_tokens_to_sample": kwargs.get("max_tokens_to_sample", 2048),
        }

    def invoke(self, prompt: str | Messages, **kwargs) -> str:
        """
        Call the model with the given prompt
        :param prompt: text to feed the model
        :param bare_bones: whether to use the 'Human: ... Assistant: ...' formatting
        :return: response
        """

        # Prompt should not be empty
        self._check_if_prompt_is_valid(prompt=prompt)

        # Call the model
        response = self._invoke(prompt=prompt)

        # Get completion
        completion = response["message"]["content"]

        # Get input and output tokens
        input_token_count = response["prompt_eval_count"]
        output_token_count = response["eval_count"]

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

    def _invoke(self, prompt: str | Messages) -> dict[str, str]:
        """
        Handle the API request to the service
        :param prompt: text to feed the model
        :return: response completion
        """

        # Format prompt, using Ollama formatting.
        # However, if we receive a string, we will format it as a single user message for ease of use.
        if isinstance(prompt, Messages):

            # Format messages
            messages = prompt.model_dump_ollama()

        elif isinstance(prompt, str):
            messages = [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
        else:
            raise ValueError("Prompt must be a string or a list of messages")

        # Payload for post request to Ollama
        payload = {
            "messages": messages,
            "options": self.model_kwargs,
            "model": self.model_id,
            "stream": False,
        }

        # Send request
        try:
            response = requests.post(url=self.chat_endpoint, json=payload)

        except Exception as e:
            raise ValueError(f"Error raised by Ollama service: {e}")

        # Check if the response is valid
        if response.status_code != 200 and response.text.__contains__("not found"):
            logger.warning(
                f"Model {self.model_id} not found in Ollama service - trying to pull it"
            )
            self._pull_model(model_id=self.model_id)

            # Retry the request
            try:
                response = requests.post(url=self.chat_endpoint, json=payload)
            except Exception as e:
                raise ValueError(f"Error raised by Ollama service: {e}")

        return response.json()

    def _pull_model(self, model_id: str) -> Response:
        """
        Pull the model from the Ollama service
        :param model_id: model name
        :return: model
        """

        # Send request
        try:
            response = requests.post(
                url=self.pull_endpoint, json={"name": model_id}, stream=False
            )

        except Exception as e:
            raise ValueError(f"Error raised by Ollama service: {e}")

        return response

    def _get_model_list(self) -> list[str]:
        """
        Get the list of available models from the Ollama service
        :return: list of model names
        """

        # Send request
        try:
            response = requests.get(url=self.tag_endpoint)

        except Exception as e:
            raise ValueError(f"Error raised by Ollama service: {e}")

        return response.json()["models"]


class Ollama(OllamaBase):
    """Generic Ollama interface"""

    def __init__(
        self, model_id: str, source: str | None = None, auto_pull: bool = True, **kwargs
    ):
        """
        Initialize the Ollama model
        :param str model_id: The model ID as defined in the Ollama service
        :param str source: An indicator of where (e.g., which feature) the model is operating from.
        :param bool auto_pull: Whether to automatically pull the model if it is not found
        :param **kwargs: Additional keyword arguments
        """

        super().__init__(source=source, auto_pull=auto_pull, **kwargs)

        self.model_id = model_id
        self.model_name = f"{model_id.capitalize()}"


class Llama3_1(Ollama):
    """Llama 3.1 model"""

    def __init__(self, source: str, **kwargs):
        """
        Initialize the Llama 3.1 model
        :param str source: An indicator of where (e.g., which feature) the model is operating from.
        :param **kwargs: Additional keyword arguments
        """

        super().__init__(model_id="llama3.1", source=source, **kwargs)


class DeepSeekR1(Ollama):
    """DeepSeek R1 model"""

    def __init__(self, source: str, **kwargs):
        """
        Initialize the DeepSeek R1 model
        :param str source: An indicator of where (e.g., which feature) the model is operating from.
        :param **kwargs: Additional keyword arguments
        """

        super().__init__(model_id="deepseek-r1", source=source, **kwargs)


if __name__ == "__main__":

    # Initialize the model
    model = Llama3_1(source="test")

    # List available models
    model_list = model._get_model_list()

    # Call the model
    response = model("Hello, how are you?")

    # Try a non-existent model
    model = Ollama(model_id="gemma2", source="test")

    # Call the model
    response_gemma = model("Hello, which model are you?")
