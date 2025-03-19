import os
import requests

from fence.embeddings.base import Embeddings

class MistralEmbeddingsBase(Embeddings):

    def __init__(
        self,
        source: str,
        full_response: bool = False,
        api_key: str | None = None,
        **kwargs
    ):
        """
        Initialize a Mistral embeddings model

        :param str source: An indicator of where (e.g., which feature) the model is operating from
        :param bool full_response: Whether to return the full response from the Mistral service
        :param **kwargs: Additional keyword arguments
        """

        super().__init__()

        self.source = source
        self.full_response = full_response
        self.api_key = api_key

        # Find API key
        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY", None)
        if not self.api_key:
            raise ValueError(
                "Mistral API key must be provided, either as an argument or in the environment variable `MISTRAL_API_KEY`"
            )
        
        self.endpoint = "https://api.mistral.ai/v1"
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
            "model": "mistral-embed",
            "input": text
        }
        
        try:
            response = requests.post(self.embedding_endpoint, headers=headers, json=data)
        
            # Check status code
            if response.status_code != 200:
                raise ValueError(f"Error raised by Mistral service: {response.text}")
        
            response = response.json()

            return response
        
        except Exception as e:
            raise ValueError(
                f"Error raised by Mistral service: {e}"
            )

class MistralEmbeddings(MistralEmbeddingsBase):
    """
    Mistral Text Embedding model
    """

    def __init__(self, source: str | None = None, **kwargs):
        """
        Initialize a Mistral Text Embedding model
        :param source: An indicator of where (e.g., which feature) the model is operating from.
        :param **kwargs: Additional keyword arguments
        """
        super().__init__(source=source, **kwargs)

        self.model_id = "mistral-embed"
        self.model_name = "Mistral Embeddings"

if __name__ == "__main__":
    # Initialize Mistral embeddings model
    mistral = MistralEmbeddings(source="mistral", model_id="mistral-small-latest")

    # Embed text
    text = "Hello, world!"
    response = mistral.embed(text=text)

    print(response)