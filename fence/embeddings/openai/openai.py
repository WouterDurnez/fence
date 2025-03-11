import os
import requests

from fence.embeddings.base import Embeddings

class OpenAIEmbeddingsBase(Embeddings):
    
    def __init__(
            self,
            source: str,
            full_response: bool = False,
            api_key: str | None = None,
            **kwargs
        ):
        """
        Initialize an OpenAI embeddings model

        :param str source: An indicator of where (e.g., which feature) the model is operating from
        :param bool full_response: Whether to return the full response from the OpenAI service
        :param **kwargs: Additional keyword arguments
        """

        super().__init__()

        self.source = source
        self.full_response = full_response
        self.api_key = api_key

        # Find API key
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", None)
        if not self.api_key:
            raise ValueError(
                "OpenAI API key must be provided, either as an argument or in the environment variable `OPENAI_API_KEY`"
            )
        
        self.endpoint = "https://api.openai.com/v1"
        self.embedding_endpoint = self.endpoint + "/embeddings"

    def embed(self, text: str, **kwargs):
        
        response = self._embed(text=text)
        embedding = response["data"][0]["embedding"]

        # Depending on the full_response flag, return either the full response or just the completion
        return response if self.full_response else embedding

    def _embed(self, text: str, **kwargs):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "input": text,
            "model": self.model_id,
            "encoding_format": "float"
        }
        
        try:
            response = requests.post(self.embedding_endpoint, headers=headers, json=data)
        
            # Check status code
            if response.status_code != 200:
                raise ValueError(f"Error raised by OpenAI service: {response.text}")
        
            response = response.json()

            return response
        
        except Exception as e:
            raise ValueError(
                f"Error raised by OpenAI service: {e}"
            )


class OpenAIEmbeddings(OpenAIEmbeddingsBase):
    """
    OpenAI Text Embedding model
    """

    def __init__(self, model_id: str, source: str | None = None, **kwargs):
        """
        Initialize an OpenAI Text Embedding model
        :param source: An indicator of where (e.g., which feature) the model is operating from.
        :param **kwargs: Additional keyword arguments
        """
        super().__init__(source=source, **kwargs)

        self.model_id = model_id
        self.model_name = f"{model_id.capitalize()}"


if __name__ == "__main__":
    # Initialize OpenAI embeddings model
    openai = OpenAIEmbeddings(source="openai", model_id="text-embedding-ada-002")

    # Embed text
    text = "Hello, world!"
    response = openai.embed(text=text)

    # Print response
    print(response)