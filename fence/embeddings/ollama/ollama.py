"""
Base class for Ollama embeddings
"""

import logging

import requests
from requests import Response

from fence.embeddings.base import Embeddings

logger = logging.getLogger(__name__)


class OllamaEmbeddingBase(Embeddings):
    """Base class for Ollama embeddings"""

    inference_type = "ollama"
    
    def __init__(
            self,
            source: str,
            full_response: bool = False,
            endpoint: str | None = None,
            **kwargs
        ):
        """
        Initialize an Ollama embeddings model

        :param str source: An indicator of where (e.g., which feature) the model is operating from
        :param str|None endpoint: The endpoint for the Ollama service
        :param **kwargs: Additional keyword arguments
        """

        super().__init__()

        self.source = source
        self.full_response = full_response

        self.endpoint = endpoint or "http://localhost:11434/api"
        self.embedding_endpoint = self.endpoint + "/embeddings"
        self.pull_endpoint = self.endpoint + "/pull"


        # Check if the endpoint is valid
        try:
            requests.get(self.embedding_endpoint)
        except requests.exceptions.ConnectionError:
            raise ValueError(
                f"No Ollama service found at {self.endpoint}. Try installing it using `brew install ollama`."
            )
        
    def embed(self, text: str, **kwargs):

        response = self._embed(text=text)
        embedding = response["embedding"]

        # Depending on the full_response flag, return either the full response or just the completion
        return response if self.full_response else embedding

    def _embed(self, text: str):
        
        payload = {
            "model" : self.model_id,
            "prompt" : text,

        }

        try:
            response = requests.post(
                url=self.embedding_endpoint, json=payload
            )

        except Exception as e:
            raise ValueError(f"Error raised by Ollama service: {e}")
        
        if response.status_code != 200 and response.text.__contains__("not found"):
            logger.warning(f"Model {self.model_id} not found in Ollama service - trying to pull it")

            self._pull_model(model_id=self.model_id)

            try:
                response = requests.post(url=self.embedding_endpoint, json=payload)
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

class OllamaEmbeddings(OllamaEmbeddingBase):
    """Generic Ollama embeddings model"""

    def __init__(self, model_id: str, source: str | None = None, auto_pull: bool = True, **kwargs):
        """
        Initialize an Ollama embeddings model
        :param str model_id: The model ID
        :param str|None source: An indicator of where (e.g., which feature) the model is operating from
        :param bool auto_pull: Whether to automatically pull the model if it is not found
        :param **kwargs: Additional keyword arguments
        """

        super().__init__(source=source, auto_pull=auto_pull, **kwargs)

        self.model_id = model_id
        self.model_name = f"{model_id.capitalize()}"

class BgeLarge(OllamaEmbeddings):
    """BGE Large embeddings model"""

    def __init__(self, source: str | None = None, **kwargs):
        """
        Initialize the BGE Large embeddings model
        :param str|None source: An indicator of where (e.g., which feature) the model is operating from
        :param **kwargs: Additional keyword arguments
        """

        super().__init__(model_id="bge-large", source=source, **kwargs)


if __name__ == "__main__":

    minillm = OllamaEmbeddings(model_id="all-minilm", source="test", full_response=True)
    print(minillm("Hello world!"))
