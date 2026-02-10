"""
Base class for Bedrock foundation models
"""

import logging
from pprint import pformat
import re
from typing import Any, get_args, get_origin, Iterator, Literal, Type

import boto3
from pydantic import AliasChoices, BaseModel, Field, ValidationError, ConfigDict
from pydantic.alias_generators import to_camel

from fence.models.base import LLM, InvokeMixin, MessagesMixin, StreamMixin
from fence.templates.messages import Messages
from fence.tools.base import BaseTool

logger = logging.getLogger(__name__)


###########################
# Structured Output Mixin #
###########################


class StructuredOutputMixin:
    """Mixin class for handling structured output with Pydantic models."""

    def _prepare_json_schema(self, schema: dict, allow_refs: bool = False):
        """
        Prepare JSON schema for Bedrock by removing all 'title' keys and optionally substituting $refs with their resolved values.

        :param schema: JSON schema to prepare
        :param allow_refs: If True, keep $refs in the schema. If False (default), resolve all $refs inline.
        """

        def _remove_titles_from_schema(schema: dict):
            if isinstance(schema, dict):
                for key in list(schema.keys()):
                    if key == "title":
                        del schema[key]
                    else:
                        _remove_titles_from_schema(schema[key])

        def _resolve_refs(schema: dict, defs: dict):
            if isinstance(schema, dict):
                for key, value in schema.items():
                    if isinstance(value, dict) and "$ref" in value:
                        ref = value["$ref"].split("/")[-1]
                        schema[key] = defs[ref]
                    elif isinstance(value, dict):
                        _resolve_refs(value, defs)

        # Recursively remove all 'title' keys from a JSON schema at all levels of nesting
        _remove_titles_from_schema(schema)

        if not allow_refs:
            # Extract all $defs from the schema and replace them with their resolved values
            defs = schema.pop("$defs", {})
            # Recursively resolve all $refs in the $defs
            _resolve_refs(defs, defs)
            # Recursively resolve all $refs in the schema
            _resolve_refs(schema, defs)

        return schema

    def _add_aliases_to_model(
        self, model: type[BaseModel], processed: set = None
    ) -> None:
        """
        Add validation aliases to a Pydantic model for Bedrock compatibility.
        Adds aliases for PascalCase, camelCase, snake_case, and other variations.
        This is necessary because Bedrock is not consistent in its case format.
        """

        def to_snake_case(name: str) -> str:
            """Convert PascalCase or camelCase to snake_case"""
            # Insert underscore before uppercase letters that follow lowercase letters
            s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
            # Insert underscore before uppercase letters that follow lowercase or digits
            s2 = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1)
            return s2.lower()

        def to_camel_case(name: str) -> str:
            """Convert snake_case or PascalCase to camelCase"""
            if "_" in name:
                # From snake_case
                components = name.split("_")
                return components[0].lower() + "".join(
                    x.title() for x in components[1:]
                )
            else:
                # From PascalCase or already camelCase
                return name[0].lower() + name[1:] if name else name

        def to_pascal_case(name: str) -> str:
            """Convert snake_case or camelCase to PascalCase"""
            if "_" in name:
                # From snake_case
                return "".join(x.title() for x in name.split("_"))
            else:
                # From camelCase or already PascalCase
                return name[0].upper() + name[1:] if name else name

        def extract_nested_models(field_type) -> list[type[BaseModel]]:
            """Extract all nested BaseModel types from a field annotation."""
            nested_models = []

            # Check if the field type itself is a BaseModel
            if isinstance(field_type, type) and issubclass(field_type, BaseModel):
                nested_models.append(field_type)
                return nested_models

            # Check for generic types like list[Model], dict[str, Model], etc.
            origin = get_origin(field_type)
            if origin is not None:
                args = get_args(field_type)
                for arg in args:
                    if isinstance(arg, type) and issubclass(arg, BaseModel):
                        nested_models.append(arg)

            return nested_models

        if processed is None:
            processed = set()

        if model in processed:
            return
        processed.add(model)

        # Set model config to populate by name (accept field name or alias)
        # Get existing config as dict, update it, and reassign
        config_dict = dict(model.model_config) if hasattr(model, "model_config") else {}
        config_dict["populate_by_name"] = True
        model.model_config = ConfigDict(**config_dict)

        # Add validation aliases to each field
        for field_name, field_info in model.model_fields.items():
            # Generate all case variations as aliases
            aliases = {
                to_snake_case(field_name),
                to_camel_case(field_name),
                to_pascal_case(field_name),
                field_name.lower(),
                field_name.upper(),
            }

            # Remove the original field name from aliases (no need to alias to itself)
            aliases.discard(field_name)

            # Set validation_alias if we have any aliases
            if aliases:
                field_info.validation_alias = AliasChoices(field_name, *aliases)

            # Recursively process nested models
            for nested_model in extract_nested_models(field_info.annotation):
                self._add_aliases_to_model(nested_model, processed)

        model.model_rebuild(force=True)

    def _setup_structured_output(
        self,
        output_structure: Type[BaseModel],
        allow_refs_in_json_schema: bool = False,
        enable_advanced_parsing: bool = True,
    ) -> None:
        """
        Configure the model for structured output.

        :param output_structure: Pydantic model class defining the output structure
        :param allow_refs_in_json_schema: If True, keep $refs in the JSON schema. If False (default), resolve all $refs inline.
        :param enable_advanced_parsing: If True (default), add validation aliases and enable schema-aware merging for multiple outputs.
        """
        if not issubclass(output_structure, BaseModel):
            raise ValueError("output_structure must be a Pydantic model class")

        try:
            # Add aliases to the model for case-insensitive field matching
            if enable_advanced_parsing:
                self._add_aliases_to_model(output_structure)

            # Convert the Pydantic model to a JSON schema
            self.json_output_schema = output_structure.model_json_schema()
            # Prepare the schema for Bedrock
            self._prepare_json_schema(
                self.json_output_schema, allow_refs=allow_refs_in_json_schema
            )
            logger.debug(f"JSON output schema: {pformat(self.json_output_schema)}")
            self.output_structure = output_structure
            logger.debug(f"Output structure: {self.output_structure}")
        except Exception as e:
            raise ValueError(f"Error parsing output structure: {e}")

        # Add structured output tool to tool config
        structured_tool = BedrockTool(
            toolSpec=BedrockToolSpec(
                name="analyze_information",
                description="Analyze the provided information.",
                inputSchema=BedrockToolInputSchema(json=self.json_output_schema),
            )
        )

        if self.toolConfig:
            self.toolConfig.tools.append(structured_tool)
        else:
            self.toolConfig = BedrockToolConfig(tools=[structured_tool])

        # Force tool use
        self.toolConfig.toolChoice = {"tool": {"name": "analyze_information"}}

    def _get_field_types(self, model: Type[BaseModel]) -> dict[str, bool]:
        """
        Inspect a Pydantic model and determine which fields are list types.

        :param model: Pydantic model class
        :return: Dictionary mapping field names to whether they are list types
        """
        from typing import get_origin

        field_is_list = {}
        for field_name, field_info in model.model_fields.items():
            annotation = field_info.annotation
            origin = get_origin(annotation)
            field_is_list[field_name] = origin is list or (
                hasattr(origin, "__name__") and "list" in origin.__name__.lower()
            )
        return field_is_list

    def _merge_field_value(
        self, output: dict, key: str, value: any, field_is_list: dict[str, bool]
    ) -> None:
        """
        Merge a field value into the output dictionary.

        :param output: Output dictionary to merge into (modified in place)
        :param key: Field name
        :param value: New value to merge
        :param field_is_list: Mapping of field names to whether they are list types
        """
        if field_is_list.get(key, False):
            # Field should be a list according to the model
            if isinstance(value, list):
                output[key].extend(value)
            else:
                output[key].append(value)
        # else: Field is a scalar, keep the first value (do nothing)

    def _initialize_field_value(
        self, output: dict, key: str, value: any, field_is_list: dict[str, bool]
    ) -> None:
        """
        Initialize a field value in the output dictionary.

        :param output: Output dictionary to initialize in (modified in place)
        :param key: Field name
        :param value: Initial value
        :param field_is_list: Mapping of field names to whether they are list types
        """
        if field_is_list.get(key, False) and not isinstance(value, list):
            # Model expects a list but we got a scalar, wrap it
            output[key] = [value]
        else:
            output[key] = value

    def _merge_tool_outputs(
        self, tool_outputs: list[dict], field_is_list: dict[str, bool]
    ) -> dict:
        """
        Merge multiple tool outputs into a single dictionary.

        List fields are merged by extending/appending.
        Scalar fields keep only the first value.

        :param tool_outputs: List of tool output dictionaries
        :param field_is_list: Mapping of field names to whether they are list types
        :return: Merged output dictionary
        """
        output = {}

        for item in tool_outputs:
            if isinstance(item, dict):
                for key, value in item.items():
                    if key in output:
                        self._merge_field_value(output, key, value, field_is_list)
                    else:
                        self._initialize_field_value(output, key, value, field_is_list)

        return output

    def _extract_structured_output(self, response_body: dict) -> dict:
        """
        Extract structured output from response body. If there are multiple outputs, they will be merged into a single dictionary.

        :param response_body: Response body from Bedrock API
        :return: Structured output as dictionary
        """

        logger.debug(f"Extracting structured output from: {response_body}")

        content_items = response_body.get("content", [{}])
        tool_uses = [tool_use for tool_use in content_items if "toolUse" in tool_use]
        tool_use_count = len(tool_uses)

        logger.debug(
            f"Content items: {len(content_items)}, Tool uses found: {tool_use_count}"
        )

        if tool_use_count == 0:
            raise ValueError("No tool use found in response")
        elif tool_use_count == 1:
            logger.debug("Single output found.")
            output = content_items[0].get("toolUse", {}).get("input", {})
        else:
            logger.debug("Multiple outputs found.")
            tool_outputs = [
                tool_use.get("toolUse", {}).get("input", {})
                for tool_use in content_items
                if "toolUse" in tool_use
            ]

            if self.enable_advanced_parsing:
                # Use schema-aware merging
                field_types = self._get_field_types(self.output_structure)
                output = self._merge_tool_outputs(tool_outputs, field_types)
            else:
                # Simple merging: just take the first output
                logger.warning(
                    "Advanced parsing is disabled. Using only the first tool output. "
                    f"Ignoring {len(tool_outputs) - 1} additional outputs."
                )
                output = tool_outputs[0] if tool_outputs else {}

        return output

    def _validate_structured_output(self, output: dict) -> None:
        """
        Validate structured output against the expected schema.

        :param output: Output dictionary to validate
        :raises ValueError: If validation fails
        """

        try:
            # Pydantic will use the validation aliases we set up
            validated = self.output_structure.model_validate(output)
            logger.debug(f"Validated structured output: {validated}")
        except ValidationError as e:
            import json

            logger.error(f"Invalid output structure: {json.dumps(output, indent=2)}")
            logger.error(f"Validation errors: {e}")
            raise ValueError(
                f"Invalid structured output from Bedrock. "
                f"Expected schema: {self.output_structure.__name__}. "
                f"Validation errors:\n{e}"
            )


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
    properties: dict[str, dict[str, Any]] = Field(
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
    toolChoice: dict | None = Field(
        None, description="The tool choice for the conversation."
    )

    def model_dump_bedrock_converse(self):
        """
        Dump the tool config in the format required by Bedrock Converse.
        """

        tools_dict = {
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
        if self.toolChoice:
            tools_dict["toolChoice"] = self.toolChoice

        return tools_dict


############################################
# Base class for Bedrock foundation models #
############################################


class BedrockBase(LLM, MessagesMixin, StreamMixin, InvokeMixin, StructuredOutputMixin):
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
        output_structure: Type[BaseModel] | None = None,
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
        :param output_structure: Pydantic model class defining the expected output structure
        :param full_response: Whether to return the full response from the Bedrock API
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

        # Output structure
        self.output_structure = None
        self.enable_advanced_parsing = kwargs.get("enable_advanced_parsing", True)
        if output_structure is not None:
            allow_refs_in_json_schema = kwargs.get("allow_refs_in_json_schema", False)
            self._setup_structured_output(
                output_structure,
                allow_refs_in_json_schema,
                self.enable_advanced_parsing,
            )

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

        # Handle runtime output_structure override
        output_structure = kwargs.pop("output_structure", None)
        if output_structure is not None:
            # Temporarily set output_structure for this invocation
            original_output_structure = self.output_structure
            original_tool_config = self.toolConfig

            # Setup structured output for this call
            self._setup_structured_output(
                output_structure,
                allow_refs_in_json_schema=False,
                enable_advanced_parsing=self.enable_advanced_parsing,
            )

            try:
                result = self._invoke(prompt=prompt, stream=False, **kwargs)
            finally:
                # Restore original state
                self.output_structure = original_output_structure
                self.toolConfig = original_tool_config

            return result

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

        # Handle runtime output_structure override
        output_structure = kwargs.pop("output_structure", None)
        if output_structure is not None:
            # Temporarily set output_structure for this invocation
            original_output_structure = self.output_structure
            original_tool_config = self.toolConfig

            # Setup structured output for this call
            self._setup_structured_output(
                output_structure,
                allow_refs_in_json_schema=False,
                enable_advanced_parsing=self.enable_advanced_parsing,
            )

            try:
                yield from self._invoke(prompt=prompt, stream=True, **kwargs)
            finally:
                # Restore original state
                self.output_structure = original_output_structure
                self.toolConfig = original_tool_config
        else:
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

            # Validate structured output if present
            if self.output_structure is not None:
                self._validate_structured_output(completion)

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

                # Handle content block start for tool use
                if "contentBlockStart" in event:
                    start = event["contentBlockStart"]["start"]
                    if "toolUse" in start:
                        logger.debug(f"Tool use start: {start["toolUse"]["name"]}")

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
                    # Handle structured output - output as string
                    elif "toolUse" in delta and self.output_structure is not None:
                        tool_use = delta["toolUse"]["input"]
                        logger.debug(f"Structured output: {tool_use}")
                        yield tool_use

                # If the event is a message stop, print the stop reason
                if "messageStop" in event:
                    logger.debug(
                        f"Message stopped. Reason: {event['messageStop']['stopReason']}"
                    )

                if "metadata" in event:
                    self._log_callback(**self._extract_metrics(event["metadata"]))

        except Exception as e:
            raise ValueError(f"Error in streaming response from Bedrock service: {e}")

    def _process_response(self, response: dict) -> str | dict:
        """
        Extracts completion text from the response.
        :param response: response from the Bedrock API
        :return: completion text or structured output
        """
        response_body = response.get("output", {}).get("message", {})

        if self.output_structure is not None:
            return self._extract_structured_output(response_body)

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
