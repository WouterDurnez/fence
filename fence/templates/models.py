"""
This module contains data models used in the API formatting.
"""

import base64
from decimal import Decimal
from mimetypes import guess_type
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

##########
# Models #
##########


class Source(BaseModel):
    """A model representing a media source supporting base64 data, raw bytes, or file path."""

    type: Literal["base64"] = "base64"
    media_type: Literal["image/jpeg", "image/png", "image/webp", "image/gif"] = Field(
        ..., description="The MIME type of the image."
    )
    data: str | bytes | None = Field(
        None, description="The image data in base64 encoding or raw bytes."
    )
    data_raw: bytes | None = Field(None, description="The image data in raw bytes.")
    file_path: str | Path | None = Field(None, description="Path to the image file.")

    @model_validator(mode="before")
    def validate_and_calculate_inputs(cls, values):

        # Count the number of provided inputs
        inputs_provided = sum(
            values.get(key) is not None for key in ["data", "data_raw", "file_path"]
        )

        # Raise error if no input or multiple inputs are provided
        if inputs_provided != 1:
            raise ValueError(
                "One and only one of 'data', 'data_raw', or 'file_path' must be provided."
            )

        # If file_path is provided, read and calculate other fields
        if values.get("file_path"):

            file_path = Path(values["file_path"])

            # Check if file exists
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            # Open the file and read the data
            with open(file_path, "rb") as file:
                raw_data = file.read()

                # Populate other fields
                values["data_raw"] = raw_data
                values["data"] = base64.b64encode(raw_data).decode("utf-8")
                values["media_type"], _ = guess_type(str(file_path))

        # If data is provided, handle both string (base64) and bytes cases
        elif values.get("data") is not None:

            # If data is a string, assume it's base64 encoded
            if isinstance(values["data"], str):

                # If data is a string, assume it's base64 encoded
                values["data_raw"] = base64.b64decode(values["data"])

                # data remains the original base64 string

            # If data is bytes, it's raw data
            elif isinstance(values["data"], bytes):

                # Move data to data_raw and calculate base64 data
                values["data_raw"] = values["data"]
                values["data"] = base64.b64encode(values["data_raw"]).decode("utf-8")

        # If only data_raw is provided, calculate base64 data
        elif values.get("data_raw"):
            values["data"] = base64.b64encode(values["data_raw"]).decode("utf-8")

        return values

    @property
    def format(self):
        return self.media_type.split("/")[-1]


class TextContent(BaseModel):
    """A model representing text content."""

    type: Literal["text"] = "text"
    text: str = Field(..., description="The text content.")


class ImageContent(BaseModel):
    """A model representing image content."""

    type: Literal["image"] = "image"
    source: Source = Field(..., description="The source of the image.")


class ToolUseBlock(BaseModel):
    """A model representing a tool use block."""

    toolUseId: str = Field(..., description="The tool use id.")
    input: dict = Field(..., description="The input to pass to the tool.")
    name: str = Field(..., description="The name of the tool the model wants to use.")

    # Loop over input values and convert to builtin types
    @field_validator("input", mode="before")
    def convert_to_builtin(cls, input):
        for parameter, value in input.items():
            if isinstance(value, Decimal):
                input[parameter] = float(value)
        return input


class ToolResultContentBlockText(BaseModel):
    """A model representing a tool result content block."""

    text: str = Field(..., description="The text content of the tool result.")


class ToolResultContentBlockJson(BaseModel):
    """A model representing a tool result content block."""

    json_field: dict = Field(
        ..., alias="json", description="The JSON content of the tool result."
    )

    model_config = {"populate_by_name": True}

    def model_dump(
        self,
        by_alias: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return super().model_dump(by_alias=by_alias, **kwargs)


ToolResultContentBlock = ToolResultContentBlockText | ToolResultContentBlockJson


class ToolResultBlock(BaseModel):
    """A model representing a tool result block."""

    content: list[ToolResultContentBlock] = Field(
        ..., description="The content of the tool result."
    )
    toolUseId: str = Field(..., description="The tool use id.")
    status: Literal["success", "error"] | None = Field(
        None, description="The status of the tool result."
    )


class ToolResultContent(BaseModel):
    """A model representing a tool result."""

    type: Literal["toolResult"] = "toolResult"
    content: ToolResultBlock


class ToolUseContent(BaseModel):
    """A model representing a tool use."""

    type: Literal["toolUse"] = "toolUse"
    content: ToolUseBlock


class ImageBlob(BaseModel):
    """A model representing image content."""

    type: Literal["image_blob"] = "image_blob"
    mime_type: str = Field(..., description="The MIME type of the image.")
    data: str = Field(..., description="The image data in base64 encoding.")


# Content can be either text or image
Content = TextContent | ImageContent | ImageBlob | ToolResultContent | ToolUseContent


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
        min_length=0,
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

                            # For images, the Converse API expects:
                            # "image": {
                            #     "format": "png | jpeg | gif | webp",
                            #     "source": {
                            #         "bytes": "image in bytes"
                            #     }
                            # }

                            package = {
                                "format": content_object.source.format,
                                "source": {
                                    "bytes": content_object.source.data_raw,
                                },
                            }

                            content.append({"image": package})
                        case "toolUse":
                            content.append(
                                {"toolUse": content_object.content.model_dump()}
                            )
                        case "toolResult":
                            content.append(
                                {
                                    "toolResult": content_object.content.model_dump(
                                        by_alias=True
                                    )
                                }
                            )
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

    def model_dump_openai(self) -> dict:
        """
        Dump the model into a dictionary for use in the OpenAI API.

        See: https://platform.openai.com/docs/guides/text-generation?text-generation-quickstart-example=image

        Example:
        [
          {
            "role": "user",
            "content": [
              {"type": "text", "text": "What's in this image?"},
              {
                "type": "image_url",
                "image_url": {
                  "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                },
              },
              {
                "type": "image_url",
                "image_url": {
                  "url": "data:image/jpeg;base64,{base64_image}" # <- to upload image
                },
              },

            ],
          }
        ],
        """

        # Go through each message and format it
        messages = []
        for message in self.messages:

            # Each message has a role (str) and content (list of content objects)
            role, content = message.role, []

            # If content is a string, append it as text
            if isinstance(message.content, str):
                content = message.content

            # If content is a list of content objects, go through each object
            elif isinstance(message.content, list):

                # Go through each content object
                for content_object in message.content:

                    # Content is either text or image, append accordingly
                    match content_object.type:
                        case "text":
                            message = {"text": content_object.text, "type": "text"}
                        case "image":
                            message = {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/{content_object.source.media_type};base64,{content_object.source.data}",
                                },
                            }
                        case "toolUse":
                            message = {
                                "type": "tool_use",
                                "tool_use": content_object.content.model_dump(),
                            }
                        case "toolResult":
                            message = {
                                "type": "tool_result",
                                "tool_result": content_object.content.model_dump(),
                            }
                        case _:
                            raise ValueError(
                                f"Content type '{content_object.type}' not recognized or supported (yet)."
                            )

                    # Append the message
                    content.append(message)

            else:
                raise TypeError(
                    "Content must be a string or a list of content objects."
                )

            # Append the message
            messages.append({"role": role, "content": content})

        # Format system message
        system = {"content": self.system, "role": "system"} if self.system else None

        # Append system message
        if system:
            messages.append(system)

        return messages

    def model_dump_anthropic(self) -> dict:
        """
        Dump the model into a dictionary for use in the Anthropic API.
        NOTE: The system prompt is not included in the messages, and will be included
        in the model class.
        """

        # Go through each message and format it
        messages = []
        for message in self.messages:

            # Each message has a role (str) and content (list of content objects)
            role, content = message.role, []

            # If content is a string, append it as text
            if isinstance(message.content, str):
                content = message.content

            # If content is a list of content objects, go through each object
            elif isinstance(message.content, list):

                # Go through each content object
                for content_object in message.content:

                    # Content is either text or image, append accordingly
                    match content_object.type:
                        case "text":
                            message = {"text": content_object.text, "type": "text"}
                        case "image":
                            message = {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/{content_object.source.media_type};base64,{content_object.source.data}",
                                },
                            }
                        case "toolUse":
                            message = {
                                "type": "tool_use",
                                "tool_use": content_object.content.model_dump(),
                            }
                        case "toolResult":
                            message = {
                                "type": "tool_result",
                                "tool_result": content_object.content.model_dump(),
                            }
                        case _:
                            raise ValueError(
                                f"Content type '{content_object.type}' not recognized or supported (yet)."
                            )

                    # Append the message
                    content.append(message)

            else:
                raise TypeError(
                    "Content must be a string or a list of content objects."
                )

            # Append the message
            messages.append({"role": role, "content": content})

        return messages

    def model_dump_gemini(self) -> list[dict]:
        """
        Dump the model into a dictionary for use in the Gemini API.

        :return: A dictionary representation of the model.
        :rtype: dict
        """

        role_map = {"user": "user", "assistant": "model"}

        # Go through each message and format it
        contents = []
        for message in self.messages:
            # Each message has a role (str) and content (list of content objects)
            role = role_map[message.role]
            parts = []

            # If content is a string, append it as text
            if isinstance(message.content, str):
                parts = {"text": message.content}

            # If content is a list of content objects, go through each object
            elif isinstance(message.content, list):

                # Go through each content object
                for content_object in message.content:

                    # Content is either text or image, append accordingly
                    match content_object.type:
                        case "text":
                            part = {"text": content_object.text}
                        case "image_blob":
                            part = {
                                "inline_data": {
                                    "mine_type": content_object.mime_type,
                                    "data": content_object.data,
                                }
                            }
                        case _:
                            raise ValueError(
                                f"Content type '{content_object.type}' not recognized or supported for Gemini (yet)."
                            )

                    # Append the message
                    parts.append(part)

            else:
                raise TypeError(
                    "Content must be a string or a list of content objects."
                )

            # Append the message
            contents.append({"role": role, "parts": parts})

        system = {"parts": {"text": self.system}} if self.system else None

        return {"contents": contents, "system_instruction": system}

    def model_dump_ollama(self) -> list[dict]:
        """
        Dump the model into a dictionary for use in the Ollama API.

        see: https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-chat-completion

        Example:
        {
            "model": "llama3.2",
            "messages": [
                {
                "role": "user",
                "content": "why is the sky blue?"
                }
            ]
        }
        """

        messages = []

        # if there is a system message, add it to the messages
        if self.system:
            messages.append({"role": "system", "content": self.system})

        for message in self.messages:
            # Each message has a role (str) and content (list of content objects)
            role, content = message.role, ""

            # If content is a string, append it as text
            if isinstance(message.content, str):
                content = message.content
            else:
                raise TypeError("Content must be a string")

            # Append the message
            messages.append({"role": role, "content": content})

        return messages

    def model_dump_mistral(self) -> list[dict]:
        """
        Dump the model into a dictionary for use in the Ollama API.

        see: https://docs.mistral.ai/api/#tag/chat/operation/chat_completion_v1_chat_completions_post

        Example:
        {
            "messages": [
                {
                "role": "user",
                "content": "why is the sky blue?"
                }
            ]
        }
        """

        messages = []

        # if there is a system message, add it to the messages
        if self.system:
            messages.append({"role": "system", "content": self.system})

        for message in self.messages:
            # Each message has a role (str) and content (list of content objects)
            role, content = message.role, ""
            # If content is a string, append it as text
            if isinstance(message.content, str):
                content = message.content
            else:
                raise TypeError("Content must be a string")

            # Append the message
            messages.append({"role": role, "content": content})

        return messages


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
