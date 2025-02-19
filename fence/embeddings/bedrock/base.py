"""
Base class for Bedrock embeddings
"""

import logging
import json

import boto3

from fence.embeddings.base import Embeddings


logger = logging.getLogger(__name__)

class BedrockEmbeddingsBase(Embeddings):
    """base class for Bedrock embeddings"""

    inference_type = "bedrock"

    def __init__(
        self,
        source: str | None = None,
        full_response: bool = False,
        **kwargs
    ):
        """
        Initialize a Bedrock model

        :param str source: An indicator of where (e.g., which feature) the model is operating from. Useful to pass to the logging callback
        :param bool full_response: whether to return the full response object or just the TEXT completion
        :param dict|None extra_tags: Additional tags to add to the logging tags
        :param **kwargs: Additional keyword arguments
        """

        super().__init__()

        self.source = source
        self.full_response = full_response

        # AWS parameters
        self.region = kwargs.get("region", "eu-central-1")

        self.model_kwargs = {
            "dimensions": kwargs.get("dimensions", 1024),
            "normalize" : kwargs.get("normalize", True),
            "embeddingTypes" : kwargs.get("embeddingTypes", ["float"]),
        }

        # Initialize the client
        self.client = boto3.client("bedrock-runtime", self.region)

    def embed(self, text: str):
        
        self._check_if_text_is_valid(text=text)
        
        response = self._embed(text)

        embedding = response["embedding"]

        # Depending on the full_response flag, return either the full response or just the completion
        return response if self.full_response else embedding
    

    def _embed(self, text: str):
        """
        Embed query text.
        :param query: Text to embed.
        :return: Embedding.
        """
        
        # Create the request for the model.
        native_request = {"inputText": text, **self.model_kwargs}

        # Convert the native request to JSON.
        request_body = json.dumps(native_request)

        try:        
            # Invoke the model with the request.
            response = self.client.invoke_model(modelId=self.model_id, body=request_body)
            # Decode the model's native response body.
            model_response = json.loads(response["body"].read())

            return model_response
        
        except Exception as e:
            raise ValueError(f"Error raised by bedrock service: {e}")

    
    @staticmethod
    def _check_if_text_is_valid(text: str):
        """
        Check if the text is a valid string
        """
        if isinstance(text, str) and not text.strip():
            raise ValueError("Prompt cannot be empty string!")
