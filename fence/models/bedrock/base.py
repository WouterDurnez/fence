"""
Base class for Bedrock foundation models
"""

import logging
from typing import Iterator, Literal

import boto3
from pydantic import BaseModel, Field

from fence.models.base import LLM, InvokeMixin, MessagesMixin, StreamMixin
from fence.templates.messages import Messages

logger = logging.getLogger(__name__)

##########
# Models #
##########


class InferenceConfig(BaseModel):
    max_tokens: int | None = Field(
        None,
        ge=1,
        description="The maximum number of tokens to allow in the generated response. "
        "The default value is the maximum allowed value for the model that you are using.",
    )
    stop_sequences: list[str] | None = Field(
        None,
        min_items=0,
        max_items=4,
        description="A list of stop sequences. A stop sequence is a sequence of characters that causes the model to stop generating the response.",
    )
    temperature: float | None = Field(
        None,
        ge=0,
        le=1,
        description="The likelihood of the model selecting higher-probability options while generating a response. "
        "A lower value makes the model more likely to choose higher-probability options, while a higher value "
        "makes the model more likely to choose lower-probability options.",
    )
    top_p: float | None = Field(
        None,
        ge=0,
        le=1,
        description="The percentage of most-likely candidates that the model considers for the next token. "
        "For example, if you choose a value of 0.8 for topP, the model selects from the top 80% of the probability "
        "distribution of tokens that could be next in the sequence.",
    )


class JSONSchema(BaseModel):
    type: Literal["object"]
    properties: dict[str, dict[str, str]]
    required: list[str]


class ToolInputSchema(BaseModel):
    json_field: JSONSchema = Field(
        ..., alias="json", description="The input schema for the tool in JSON format."
    )


class ToolSpec(BaseModel):
    name: str = Field(
        ..., min_length=1, max_length=64, description="The name for the tool."
    )
    description: str | None = Field(
        None, min_length=1, description="The description for the tool."
    )
    inputSchema: ToolInputSchema = Field(
        ..., description="The input schema for the tool in JSON format."
    )


class Tool(BaseModel):
    toolSpec: ToolSpec | None = Field(..., description="The tool specification.")


class ToolConfig(BaseModel):
    tools: list[Tool] = Field(
        ..., min_length=1, description="The tools to use for the conversation."
    )


############################################
# Base class for Bedrock foundation models #
############################################


class BedrockBase(LLM, MessagesMixin, StreamMixin, InvokeMixin):
    """Base class for Bedrock foundation models"""

    inference_type = "bedrock"

    ###############
    # Constructor #
    ###############

    def __init__(
        self,
        source: str | None = None,
        inferenceConfig: InferenceConfig | dict | None = None,
        toolConfig: ToolConfig | dict | None = None,
        full_response: bool = False,
        **kwargs,
    ):
        """
        Initialize a Bedrock model

        :param str source: An indicator of where (e.g., which feature) the model is operating from. Useful to pass to the logging callback
        :param **kwargs: Additional keyword arguments
        """
        super().__init__(**kwargs)
        self.source = source

        # Config
        self.inferenceConfig = (
            InferenceConfig(**inferenceConfig)
            if isinstance(inferenceConfig, dict)
            else inferenceConfig
        )
        self.toolConfig = (
            ToolConfig(**toolConfig) if isinstance(toolConfig, dict) else toolConfig
        )
        self.full_response = full_response

        # AWS
        self.region = kwargs.get("region", "eu-central-1")
        self.client = boto3.client("bedrock-runtime", self.region)

    def invoke(self, prompt: str | Messages, **kwargs) -> str:
        """
        Call the model with the given prompt
        :param prompt: text to feed the model
        :param **kwargs: Additional keyword arguments to override the default model kwargs
        :return: response
        """
        if "full_response" in kwargs:
            self.full_response = kwargs["full_response"]
        return self._invoke(prompt=prompt, stream=False, **kwargs)

    def stream(self, prompt: str | Messages, **kwargs) -> Iterator[str]:
        """
        Stream the model response with the given prompt.
        :param prompt: text to feed the model
        :param **kwargs: Additional keyword arguments to override the default model kwargs
        :return: stream of responses
        """
        if "full_response" in kwargs:
            self.full_response = kwargs["full_response"]
        yield from self._invoke(prompt=prompt, stream=True, **kwargs)

    def _invoke(self, prompt: str | Messages, stream: bool, **kwargs):
        """
        Centralized method to handle both invoke and stream.
        """
        # Check if the prompt is valid
        self._check_if_prompt_is_valid(prompt=prompt)

        # Prepare the request parameters with override kwargs
        invoke_params = self._prepare_request(prompt)

        # Invoke the model
        if stream:
            return self._handle_stream(invoke_params=invoke_params)
        else:
            return self._handle_invoke(invoke_params=invoke_params)

    def _prepare_request(self, prompt: str | Messages) -> dict:
        """
        Prepares the request parameters for the Bedrock API.

        :param prompt: text or Messages object to format
        :return: request parameters dictionary
        """
        messages = self._format_prompt(prompt)

        # Prepare the request parameters
        invoke_params = {
            "modelId": self.model_id,
            "messages": messages["messages"],
        }
        if self.inferenceConfig:
            invoke_params["inferenceConfig"] = self.inferenceConfig.model_dump()
        if self.toolConfig:
            invoke_params["toolConfig"] = self.toolConfig.model_dump(by_alias=True)
        if "system" in messages:
            invoke_params["system"] = messages["system"]
        return invoke_params

    def _handle_invoke(self, invoke_params: dict) -> str | dict:
        """
        Handles synchronous invocation.

        :param invoke_params: parameters for the Bedrock API
        :return: completion text or full response object if full_response is True
        """
        try:
            # Call the Bedrock API
            response = self.client.converse(**invoke_params)

            # Extract and log metrics regardless of full_response setting
            metrics = self._extract_metrics(response.get("usage", {}))
            self._log_callback(**metrics)

            # Return the full response if requested, without processing
            if self.full_response:
                return response

            # Process the response for completion text
            completion = self._process_response(response)
            return completion

        except Exception as e:
            raise ValueError(f"Error raised by bedrock service: {e}")

    def _handle_stream(self, invoke_params: dict) -> Iterator[str | dict]:
        """
        Handles streaming invocation.

        :param invoke_params: parameters for the Bedrock API
        :return: stream of responses or full response objects if full_response is True
        """
        try:
            # Call the Bedrock streaming API
            response = self.client.converse_stream(**invoke_params)

            # Process the streaming response
            stream = response.get("stream")
            if not stream:
                raise ValueError("No stream found in response")

            # Stream the response
            for event in stream:
                # If full_response is True, yield the entire event
                if self.full_response:
                    yield event
                    continue

                # If the event is a message start, print the role
                if "messageStart" in event:
                    logger.debug(
                        f"Message started. Role: {event['messageStart']['role']}"
                    )

                # Handle content blocks - both text and tool calls
                if "contentBlockDelta" in event:
                    delta = event["contentBlockDelta"]["delta"]
                    if "text" in delta:
                        text = delta["text"]
                        logger.debug(f"Streaming chunk: {text}")
                        yield text
                    elif "toolCall" in delta:
                        tool_call = delta["toolCall"]
                        logger.debug(f"Tool call: {tool_call}")
                        yield str(tool_call)

                # If the event is a message stop, print the stop reason
                if "messageStop" in event:
                    logger.debug(
                        f"Message stopped. Reason: {event['messageStop']['stopReason']}"
                    )

                if "metadata" in event:
                    self._log_callback(**self._extract_metrics(event["metadata"]))

        except Exception as e:
            raise ValueError(f"Error in streaming response from Bedrock service: {e}")

    def _process_response(self, response: dict) -> str:
        """
        Extracts completion text from the response.
        """
        response_body = response.get("output", {}).get("message", {})
        completion = response_body.get("content", [{}])[0].get("text", "")
        return completion

    def _extract_metrics(self, metadata: dict) -> dict:
        """
        Extracts relevant metrics from the metadata.
        """
        return {
            "input_token_count": metadata.get("inputTokens", 0),
            "output_token_count": metadata.get("outputTokens", 0),
            "total_token_count": metadata.get("totalTokens", 0),
            "latency_ms": metadata.get("metrics", {}).get("latencyMs"),
        }

    def _format_prompt(self, prompt: str | Messages) -> dict:
        """
        Format the prompt for the Bedrock API

        :param prompt: text or Messages object to format
        :return: formatted messages dictionary
        """
        # Format prompt: Claude3 models expect a list of messages, with content, roles, etc.
        # However, if we receive a string, we will format it as a single user message for ease of use.
        if isinstance(prompt, Messages):
            messages = prompt.model_dump_bedrock_converse()
            # Strip `type` key from each message
            messages["messages"] = [
                {k: v for k, v in message.items() if k != "type"}
                for message in messages["messages"]
            ]
        elif isinstance(prompt, str):
            messages = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "text": prompt,
                            }
                        ],
                    }
                ]
            }
        else:
            raise ValueError("Prompt must be a string or a list of messages")
        return messages


if __name__ == "__main__":
    tool_config = {
        "tools": [
            {
                "toolSpec": {
                    "name": "top_song",
                    "description": "Get the most popular song played on a radio station.",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "sign": {
                                    "type": "string",
                                    "description": "The call sign for the radio station for which you want the most popular song. Example calls signs are WZPZ, and WKRP.",
                                }
                            },
                            "required": ["sign"],
                        }
                    },
                }
            }
        ]
    }
    # Try to parse the tool config
    try:
        tool_config = ToolConfig(**tool_config)
    except Exception as e:
        print(e)
