"""
This module contains data models used in the API formatting.
"""

from typing import Literal, Optional

from pydantic import BaseModel

##########
# Models #
##########


class Source(BaseModel):
    """A model representing a media source. A source is always base64 encoded."""

    type: Literal["base64"] = "base64"
    media_type: Literal["image/jpeg", "image/png", "image/webp", "image/gif"]
    data: str


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
