"""
This module contains data models used in the API formatting.
"""

import base64
from mimetypes import guess_type
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, model_validator

##########
# Models #
##########


class Source(BaseModel):
    """A model representing a media source. Can be base64 encoded data or a URL."""

    type: Literal["base64"] = "base64"
    media_type: Literal["image/jpeg", "image/png", "image/webp", "image/gif"]
    data: str | None = None
    _file_path: str | Path | None = None

    # Validate that only one of 'data' or 'url' is provided
    @model_validator(mode="before")
    def handle_data_or_url(cls, values):

        # Extract data and file_path values
        data, file_path = values.get("data"), values.get("_file_path")

        # Cannot provide both data and file_path
        if data and file_path:
            raise ValueError("Only one of 'data' or 'file_path' should be provided.")

        # Must provide either data or file_path
        if not data and not file_path:
            raise ValueError("One of 'data' or 'file_path' must be provided.")

        # If a file path is provided, check if it is a valid local path
        if file_path:
            with open(file_path, "rb") as file:
                image_data = file.read()
                values["data"] = base64.b64encode(image_data).decode("utf-8")
                values["media_type"], _ = guess_type(file_path)

        return values


class TextContent(BaseModel):
    """A model representing text content."""

    type: Literal["text"] = "text"
    text: str


class ImageContent(BaseModel):
    """A model representing image content."""

    type: Literal["image"] = "image"
    source: Source


# Content can be either text or image
Content = TextContent | ImageContent


class Message(BaseModel):
    """A base model representing a message."""

    role: Literal["user", "assistant"]
    content: list[Content] | str


class Messages(BaseModel):
    """A model representing a collection of messages."""

    messages: list[Message]
    system: Optional[str] = None
