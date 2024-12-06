"""
This module contains data models used in the API formatting.
"""

import base64
from mimetypes import guess_type
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

##########
# Models #
##########


class Source(BaseModel):
    """A model representing a media source. Can be base64 encoded data or a URL."""

    type: Literal["base64"] = "base64"
    media_type: Literal["image/jpeg", "image/png", "image/webp", "image/gif"] = Field(
        ..., description="The MIME type of the image."
    )
    data: str | None = Field(None, description="The base64 encoded data.")
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
    text: str = Field(..., description="The text content.")


class ImageContent(BaseModel):
    """A model representing image content."""

    type: Literal["image"] = "image"
    source: Source = Field(..., description="The source of the image.")


# Content can be either text or image
Content = TextContent | ImageContent


class Message(BaseModel):
    """A base model representing a message."""

    role: Literal["user", "assistant"] = Field(
        ..., description="The role of the message."
    )
    content: list[Content] | str = Field(
        ...,
        description="The content of the message.",
    )


class Messages(BaseModel):
    """A model representing a collection of messages."""

    messages: list[Message] = Field(
        ...,
        description="List of messages.",
        min_length=1,
    )
    system: str | None = Field(
        None,
        description="System message.",
    )

    # API formatting #

    def model_dump_bedrock_converse(self) -> dict:
        """
        Dump the model into a dictionary for use in the Bedrock Converse API.

        See `https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html#bedrock-runtime_Converse-request-system` for more information.

        :param bool exclude_none: Whether to exclude None values.
        :return: A dictionary representation of the model.
        :rtype: dict
        """

        # Go through each message and format it
        messages = []
        for message in self.messages:

            # Each message has a role (str) and content (list of content objects)
            role, content = message.role, []

            # If content is a string, append it as text
            if isinstance(message.content, str):
                content.append({"text": message.content})

            # If content is a list of content objects, go through each object
            elif isinstance(message.content, list):

                # Go through each content object
                for content_object in message.content:

                    # Content is either text or image, append accordingly
                    match content_object.type:
                        case "text":
                            content.append({"text": content_object.text})
                        case "image":
                            content.append({"image": content_object.source.dict()})
                        case _:
                            raise ValueError(
                                f"Content type '{content_object.type}' not recognized or supported (yet)."
                            )

            else:
                raise TypeError(
                    "Content must be a string or a list of content objects."
                )

            # Append the message
            messages.append({"role": role, "content": content})

        system = [{"text": self.system}] if self.system else None

        return {"messages": messages, "system": system}


if __name__ == "__main__":

    # Example messages
    messages = Messages(
        messages=[
            Message(
                role="user",
                content="\nThis is the text:\n\n```\nMixed martial arts (MMA)[a] is a full-contact fighting sport based o...s match was the first popular fight which showcased the power of such low kicks to a predominantly Western audience.[46]```\n",
            )
        ],
        system="This is the system message.",
    )

    # Model dump regular
    from pprint import pprint

    pprint(messages.model_dump())

    # Model dump converse API
    from pprint import pprint

    pprint(messages.model_dump_bedrock_converse())
