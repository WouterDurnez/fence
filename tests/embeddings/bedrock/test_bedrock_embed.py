import pytest

from fence.embeddings.bedrock.base import BedrockEmbeddingsBase
from fence.embeddings.bedrock.titan import TitanEmbeddingsV2


class MockBedrockEmbeddingsBase(BedrockEmbeddingsBase):
    def _embed(self, text):
        return {"embedding": [0.1, 0.2, 0.3]}


class TestBedrockEmbeddingsBase:
    @pytest.fixture
    def bedrock_base(self):
        return MockBedrockEmbeddingsBase()

    @pytest.fixture
    def bedrock_base_full_response(self):
        return MockBedrockEmbeddingsBase(full_response=True)

    def test_initialization(self, bedrock_base):
        assert bedrock_base.source is None
        assert bedrock_base.full_response is False
        assert bedrock_base.region == "eu-central-1"
        assert bedrock_base.model_kwargs == {
            "dimensions": 1024,
            "normalize": True,
            "embeddingTypes": ["float"],
        }

    def test_embed(self, bedrock_base):
        embedding = bedrock_base.embed("Test prompt")
        assert embedding == [0.1, 0.2, 0.3]

    def test_embed_full_response(self, bedrock_base_full_response):
        response = bedrock_base_full_response.embed("Test prompt")
        assert response == {"embedding": [0.1, 0.2, 0.3]}

    def test_bedrock_base_invoke_with_empty_prompt(self, bedrock_base):
        """
        Test case for the invoke method of the BedrockBase class with an empty prompt.
        This test checks if the invoke method raises a ValueError when the prompt is empty.
        """
        with pytest.raises(ValueError):
            bedrock_base.embed(text="")


class MockTitanEmbeddingsV2(TitanEmbeddingsV2):
    def _embed(self, text):
        return {
            "embedding": [
                -0.03553599491715431,
                0.003311669221147895,
                -0.04013663902878761,
            ],
            "embeddingsByType": {
                "float": [
                    -0.03553599491715431,
                    0.003311669221147895,
                    -0.04013663902878761,
                ],
                "binary": [0, 1, 0],
            },
            "inputTextTokenCount": 10,
        }


class TestBedrockTitanEmbeddings:
    @pytest.fixture
    def titan_base(self):
        return TitanEmbeddingsV2()

    def test_initialization(self, titan_base):
        assert titan_base.model_id == "amazon.titan-embed-text-v2:0"
        assert titan_base.model_name == "Titan Text Embeddings V2"

    def test_embed(self, titan_base):
        titan = MockTitanEmbeddingsV2()
        embedding = titan.embed("Test prompt")
        assert embedding == [
            -0.03553599491715431,
            0.003311669221147895,
            -0.04013663902878761,
        ]

    def test_embed_full_response(self, titan_base):
        titan = MockTitanEmbeddingsV2(full_response=True)
        embedding = titan.embed("Test prompt")

        assert embedding["embedding"] == [
            -0.03553599491715431,
            0.003311669221147895,
            -0.04013663902878761,
        ]
        assert embedding["embeddingsByType"].keys() == {"float", "binary"}
        assert embedding["inputTextTokenCount"] == 10

    def test_titan_base_invoke_with_empty_prompt(self, titan_base):
        """
        Test case for the invoke method of the TitanEmbeddingsV2 class with an empty prompt.
        This test checks if the invoke method raises a ValueError when the prompt is empty.
        """
        with pytest.raises(ValueError):
            titan_base.embed(text="")
