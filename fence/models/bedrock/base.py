"""
Base class for Bedrock foundation models
"""

import logging
from pprint import pformat
from typing import Iterator, Literal

import boto3
from pydantic import BaseModel, Field
from pydantic.alias_generators import to_camel

from fence.models.base import LLM, InvokeMixin, MessagesMixin, StreamMixin
from fence.templates.messages import Messages
from fence.tools.base import BaseTool

logger = logging.getLogger(__name__)

##########
# Models #
##########


class BedrockInferenceConfig(BaseModel):
    max_tokens: int | None = Field(
        None,
        ge=1,
        description="The maximum number of tokens to allow in the generated response. "
        "The default value is the maximum allowed value for the model that you are using.",
    )
    stop_sequences: list[str] | None = Field(
        None,
        min_length=0,
        max_length=4,
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

    model_config = {
        "populate_by_name": True,
        "alias_generator": to_camel,
    }


class BedrockJSONSchema(BaseModel):
    """
    JSON schema for the tool input.

    Example:
    ```json
    {
        "type": "object",
        "properties": {
            "sign": {
            "type": "string",
            "description": "The call sign for the radio station for which you want the most popular song. Example calls signs are WZPZ, and WKRP.",
        }
        },
        "required": ["sign"],
    }
    ```
    """

    type: Literal["object"] = Field(..., description="The type of the schema.")
    properties: dict[str, dict[str, str]] = Field(
        ..., description="The properties of the schema."
    )
    required: list[str] = Field(
        [], description="The required properties of the schema."
    )


class BedrockToolInputSchema(BaseModel):
    json_field: BedrockJSONSchema = Field(
        ..., alias="json", description="The input schema for the tool in JSON format."
    )


class BedrockToolSpec(BaseModel):
    name: str = Field(
        ..., min_length=1, max_length=64, description="The name for the tool."
    )
    description: str | None = Field(
        None, min_length=1, description="The description for the tool."
    )
    inputSchema: BedrockToolInputSchema = Field(
        ..., description="The input schema for the tool in JSON format."
    )


class BedrockTool(BaseModel):
    toolSpec: BedrockToolSpec | None = Field(..., description="The tool specification.")


class BedrockToolConfig(BaseModel):
    tools: list[BedrockTool] = Field(
        ..., min_length=1, description="The tools to use for the conversation."
    )

    def model_dump_bedrock_converse(self):
        """
        Dump the tool config in the format required by Bedrock Converse.
        """
        return {
            "tools": [
                {
                    "toolSpec": {
                        "name": tool.toolSpec.name,
                        "description": tool.toolSpec.description,
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": tool.toolSpec.inputSchema.json_field.properties,
                                "required": tool.toolSpec.inputSchema.json_field.required,
                            }
                        },
                    }
                }
                for tool in self.tools
            ]
        }


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
        inferenceConfig: BedrockInferenceConfig | dict | None = None,
        toolConfig: BedrockToolConfig | dict | list[BedrockTool] | None = None,
        additionalModelRequestFields: dict | None = None,
        full_response: bool = False,
        cross_region: Literal["us", "eu"] | None = None,
        **kwargs,
    ):
        """
        Initialize a Bedrock model

        :param source: An indicator of where (e.g., which feature) the model is operating from. Useful to pass to the logging callback
        :param cross_region: Region prefix for model ID, either "us" or "eu". Use this to switch to an inference profile, for cross-region inference.
        :param inferenceConfig: Inference configuration for the model
        :param toolConfig: Tool configuration for the model
        :param additionalModelRequestFields: Additional model request fields to pass to the Bedrock API
        :param full_response: Whether to return the full response from the Bedrock API
        :param **kwargs: Additional keyword arguments
        """
        super().__init__(**kwargs)
        self.source = source

        # Config
        self.inferenceConfig = (
            BedrockInferenceConfig(**inferenceConfig)
            if isinstance(inferenceConfig, dict)
            else inferenceConfig
        )

        # Handle toolConfig initialization with proper match/case syntax
        self.toolConfig = None
        if toolConfig:
            match toolConfig:
                case BedrockToolConfig():
                    self.toolConfig = toolConfig
                case dict():
                    self.toolConfig = BedrockToolConfig(**toolConfig)
                case list():
                    # Check if all items in the list are BedrockTool
                    if all(isinstance(item, BedrockTool) for item in toolConfig):
                        self.toolConfig = BedrockToolConfig(tools=toolConfig)
                    elif all(isinstance(item, BaseTool) for item in toolConfig):
                        self.toolConfig = BedrockToolConfig(
                            tools=[
                                BedrockTool(**item.model_dump_bedrock_converse())
                                for item in toolConfig
                            ]
                        )
                    else:
                        raise ValueError(
                            "All items in the toolConfig list must be BedrockTool or BaseTool objects"
                        )
                case _:
                    raise ValueError(f"Invalid toolConfig type: {type(toolConfig)}")

        # Additional model request fields
        self.additionalModelRequestFields = additionalModelRequestFields

        # Full response
        self.full_response = full_response

        # AWS
        self.region = kwargs.get("region", "eu-central-1")
        self.client = boto3.client("bedrock-runtime", self.region)
        logger.debug(f"Initialized Bedrock client in region: {self.region}")

        # Update model ID with cross-region prefix (us or eu) if specified
        if cross_region:
            if cross_region not in {"us", "eu"}:
                raise ValueError(
                    f"Invalid cross_region value: {cross_region}. Should be either 'us' or 'eu'."
                )
            else:
                self.model_id = f"{cross_region}.{self.model_id}"

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
            invoke_params["inferenceConfig"] = self.inferenceConfig.model_dump(
                by_alias=True, exclude_none=True
            )
        if self.toolConfig:
            invoke_params["toolConfig"] = self.toolConfig.model_dump(
                by_alias=True, exclude_none=True
            )
        if self.additionalModelRequestFields:
            invoke_params["additionalModelRequestFields"] = (
                self.additionalModelRequestFields
            )
        if "system" in messages and messages["system"] is not None:
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
            logger.debug(
                f"Invoking Bedrock model with params: {pformat(invoke_params)}"
            )
            response = self.client.converse(**invoke_params)
            logger.debug(f"Response: {pformat(response)}")

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
    # tool_config = {
    #     "tools": [
    #         {
    #             "toolSpec": {
    #                 "name": "top_song",
    #                 "description": "Get the most popular song played on a radio station.",
    #                 "inputSchema": {
    #                     "json": {
    #                         "type": "object",
    #                         "properties": {
    #                             "sign": {
    #                                 "type": "string",
    #                                 "description": "The call sign for the radio station for which you want the most popular song. Example calls signs are WZPZ, and WKRP.",
    #                             }
    #                         },
    #                         "required": ["sign"],
    #                     }
    #                 },
    #             }
    #         }
    #     ]
    # }
    # # Try to parse the tool config
    # try:
    #     tool_config = BedrockToolConfig(**tool_config)
    # except Exception as e:
    #     print(e)

    inferenceConfig = BedrockInferenceConfig(maxTokens=1, temperature=1, topP=1)

    print(inferenceConfig.model_dump())
