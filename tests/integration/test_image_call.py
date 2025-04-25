"""
Integration test for the image models.
"""

import base64
import os
from pathlib import Path

import pytest

from fence.links import Link
from fence.models.openai import GPT4omini
from fence.templates.messages import MessagesTemplate
from fence.templates.models import ImageContent, Message, Messages, Source

# Check if OpenAI API key is present
has_openai_api_key = os.environ.get("OPENAI_API_KEY") is not None


# Fixture to set up dependencies
@pytest.fixture
def setup_environment():
    model = GPT4omini()
    pwd = Path(__file__).absolute()
    docs = pwd.parent.parent.parent / "docs"
    logo_path = docs / "logo.png"

    # Ensure the logo file exists
    if not logo_path.is_file():
        pytest.fail(f"Test setup failed: File {logo_path} not found.")

    # Read and prepare the image source
    with open(logo_path, "rb") as f:
        file_content = f.read()

    source = Source(
        data=base64.b64encode(file_content).decode("utf-8"), media_type="image/png"
    )
    return model, source


# Test function
@pytest.mark.skipif(
    not has_openai_api_key, reason="OpenAI API key not found in environment"
)
def test_image_processing_response(setup_environment):
    """Test that the response is successfully returned without error."""
    model, source = setup_environment

    # Create image content and messages
    image_content = ImageContent(source=source)
    message = Message(role="user", content=[image_content])
    messages = Messages(system="Describe the image", messages=[message])

    # Create a template and link
    template = MessagesTemplate(source=messages)
    link = Link(template=template, model=model, name="image_processor")

    # Run the link and get the response
    response = link.run()["state"]

    # Assertions
    assert response is not None, "Response should not be None"
