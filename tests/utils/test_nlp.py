"""
NLP utility tests
"""

import os

import pytest

from fence.models.openai.gpt import GPT4omini
from fence.utils.nlp import LLMHelper, TextChunker, get_first_n_words, get_word_count

# Check if OpenAI API key is present
has_openai_api_key = os.environ.get("OPENAI_API_KEY") is not None

##############
# Test Utils #
##############


def test_get_first_n_words():
    """
    Test the `get_first_n_words` function.
    Ensure that the correct number of words is returned from the start of a given string.
    Test edge cases such as requesting 0 words and providing an empty string.
    """
    text = "The quick brown fox jumps over the lazy dog"
    assert get_first_n_words(text, 4) == "The quick brown fox"
    assert get_first_n_words(text, 1) == "The"
    assert get_first_n_words(text, 0) == ""
    assert get_first_n_words("", 5) == ""


def test_get_word_count():
    """
    Test the `get_word_count` function.
    Verify the function correctly counts the number of words in a string,
    handling edge cases like empty strings, single words, and strings with multiple spaces.
    """
    assert get_word_count("The quick brown fox") == 4
    assert get_word_count("") == 0
    assert get_word_count("word") == 1
    assert get_word_count("Multiple    spaces   between") == 3


####################
# Test TextChunker #
####################


@pytest.fixture
def sample_text():
    """
    Provide a sample long text for chunking tests.
    This text is repeated multiple times to ensure it's large enough to be split into chunks.
    """
    return (
        "This is a test text that we will use to verify the chunking functionality. "
        * 10
    )


def test_text_chunker_init():
    """
    Test that the `TextChunker` class initializes correctly.
    Verify that the chunk size and overlap values are set as expected,
    and the initial text is None by default.
    """
    chunker = TextChunker(chunk_size=100, overlap=0.1)
    assert chunker.chunk_size == 100
    assert chunker.overlap == 0.1
    assert chunker.text is None


def test_text_chunker_with_no_overlap():
    """
    Test that `TextChunker` splits text correctly when there is no overlap.
    Verify that chunks do not contain overlapping content between consecutive chunks.
    """
    chunker = TextChunker(chunk_size=50, overlap=0)
    text = "This is a test text that needs to be split into chunks without overlap"
    chunks = chunker.split_text(text)

    # Ensure no overlap between consecutive chunks
    for i in range(len(chunks) - 1):
        assert chunks[i][-10:] not in chunks[i + 1][:10]


def test_text_chunker_with_overlap():
    """
    Test that `TextChunker` splits text correctly when there is a defined overlap.
    Verify that some overlap occurs between consecutive chunks based on the overlap setting.
    """
    chunker = TextChunker(chunk_size=50, overlap=0.1)
    text = "This is a test text that needs to be split into chunks with some overlap between them"
    chunks = chunker.split_text(text)

    # Ensure overlap exists between consecutive chunks
    for i in range(len(chunks) - 1):
        end_of_first = chunks[i][-10:]
        start_of_second = chunks[i + 1][: len(end_of_first)]
        assert len(set(end_of_first.split()) & set(start_of_second.split())) > 0


def test_text_chunker_with_small_text():
    """
    Test that `TextChunker` handles cases where the text is smaller than the chunk size.
    Verify that the text is returned as a single chunk if it is smaller than the chunk size.
    """
    chunker = TextChunker(chunk_size=1000, overlap=0.1)
    text = "Small text"
    chunks = chunker.split_text(text)
    assert len(chunks) == 1
    assert chunks[0] == text


##################
# Test LLMHelper #
##################


@pytest.fixture
def llm_helper():
    """
    Fixture to create an instance of LLMHelper for use in multiple tests.
    This avoids redundant initialization across tests.
    """
    return LLMHelper()


def test_check_relevancy(mocker, llm_helper):
    """
    Test the `check_relevancy` method of LLMHelper.
    Mock the LLM's `Link.run` method to simulate a successful relevancy check.
    Verify that the function returns the correct result based on the mock response.
    """
    # Mock the Link.run method
    mock_run = mocker.patch("fence.links.Link.run", return_value={"is_relevant": True})

    text = "The sky is blue due to Rayleigh scattering."
    topic = "The sky is blue"
    result = llm_helper.check_relevancy(text, topic)

    assert result is True
    mock_run.assert_called_once()


def test_remove_text_references(mocker, llm_helper):
    """
    Test the `remove_text_references` method of LLMHelper.
    Mock the LLM's `Link.run` method to simulate the removal of text references.
    Verify that the function correctly cleans the text by removing references as indicated by the mock response.
    """
    # Mock the Link.run method
    mock_run = mocker.patch(
        "fence.links.Link.run",
        return_value={"cleaned_text": "The sky is blue due to Rayleigh scattering."},
    )

    text = "According to the text, the sky is blue due to Rayleigh scattering."
    result = llm_helper.remove_text_references(text)

    assert result == "The sky is blue due to Rayleigh scattering."
    mock_run.assert_called_once()


@pytest.mark.skipif(
    not has_openai_api_key, reason="OpenAI API key not found in environment"
)
def test_check_relevancy_using_model(mocker, llm_helper):
    """
    Test `check_relevancy` method using an actual model instance.
    This test runs the method without mocking, using the `GPT4omini` model to check text relevancy.
    """
    llm_helper = LLMHelper(model=GPT4omini())
    result = llm_helper.check_relevancy(
        "The sky is blue due to Rayleigh scattering.", "The sky is blue"
    )
    assert result is True

    result = llm_helper.check_relevancy(
        "The sky is blue due to Rayleigh scattering.", "Pizza is delicious"
    )
    assert result is False
