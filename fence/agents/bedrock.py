"""
Bedrock agent class that uses native tool calling and streaming capabilities.
"""

import json
import logging
import re
from typing import Any, Callable, Generator, Iterator, List, Union

from pydantic import BaseModel, field_validator

from fence.agents.base import AgentLogType, BaseAgent
from fence.memory.base import BaseMemory, FleetingMemory
from fence.models.base import LLM
from fence.models.bedrock.base import BedrockTool, BedrockToolConfig
from fence.streaming.base import StreamHandler
from fence.templates.models import Messages
from fence.tools.base import BaseTool

logger = logging.getLogger(__name__)


HandlerType = Union[Callable, List[Callable]]


class EventHandler(BaseModel):
    """Event handler for the BedrockAgent.

    :param on_tool_use: Called when the agent uses a tool
    :param on_thinking: Called when the agent is thinking
    :param on_answer: Called when the agent provides text answer chunks
    """

    on_tool_use: HandlerType | None = None
    on_thinking: HandlerType | None = None
    on_answer: HandlerType | None = None

    @field_validator("*", mode="before")
    def validate_handlers(cls, value, info):
        """Validate that handler functions are callables or lists of callables."""
        if value is None:
            return None

        field_name = info.field_name

        def validate_callable_signature(handler, field_name):
            import inspect

            # Check parameter count requirements by handler type
            sig = inspect.signature(handler)
            param_count = len(
                [
                    p
                    for p in sig.parameters.values()
                    if p.default == inspect.Parameter.empty
                ]
            )

            if field_name == "on_tool_use" and param_count < 3:
                raise ValueError(
                    "on_tool_use handler must accept at least 3 parameters: tool_name, parameters, result"
                )
            elif field_name in ("on_thinking", "on_answer") and param_count < 1:
                raise ValueError(
                    f"{field_name} handler must accept at least 1 parameter: text"
                )

        if isinstance(value, list):
            for handler in value:
                if not callable(handler):
                    raise ValueError(f"Handler {handler} is not callable")
                validate_callable_signature(handler, field_name)
            return value
        elif callable(value):
            validate_callable_signature(value, field_name)
            return value
        else:
            raise ValueError(f"Handler {value} is not callable")


class BedrockAgent(BaseAgent):
    """
    Bedrock agent that uses native tool calling and streaming capabilities.
    """

    _BASE_SYSTEM_MESSAGE = """
You are a helpful assistant. You can think in <thinking> tags, and provide an answer in <answer> tags.
"""

    _THINKING_PATTERN = re.compile(r"<thinking>(.*?)</thinking>", re.DOTALL)
    _ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

    def __init__(
        self,
        identifier: str | None = None,
        model: LLM = None,
        description: str | None = None,
        memory: BaseMemory | None = None,
        environment: dict | None = None,
        prefill: str | None = None,
        log_agentic_response: bool = True,
        are_you_serious: bool = False,
        tools: list[BaseTool] | None = None,
        system_message: str | None = None,
        event_handlers: EventHandler | dict[str, HandlerType] | None = None,
    ):
        """
        Initialize the BedrockAgent object.

        :param identifier: An identifier for the agent
        :param model: A Bedrock LLM model object
        :param description: A description of the agent
        :param memory: A memory object to store messages and system messages
        :param environment: A dictionary of environment variables to pass to tools
        :param prefill: A string to prefill the memory with
        :param log_agentic_response: A flag to determine if the agent's responses should be logged
        :param are_you_serious: A flag to determine if the log message should be printed in a frivolous manner
        :param tools: A list of tools to make available to the agent
        :param system_message: A system message to set for the agent
        :param event_handlers: Event handlers for different agent events (on_tool_use, on_thinking, on_answer)
        """

        # We will need the full response to be set to True
        model.full_response = True

        super().__init__(
            identifier=identifier,
            model=model,
            description=description,
            memory=memory,
            environment=environment,
            prefill=prefill,
            log_agentic_response=log_agentic_response,
            are_you_serious=are_you_serious,
        )

        # Set up the system message
        if system_message:
            # Combine with the default system message
            self._system_message = f"{self._BASE_SYSTEM_MESSAGE}\n\n{system_message}"
        else:
            # Use just the default system message
            self._system_message = self._BASE_SYSTEM_MESSAGE

        # Set the system message in memory
        self.memory.set_system_message(self._system_message)

        # Set the tools
        self.tools = tools or []

        # Update with user-provided event handlers
        self._set_event_handlers(event_handlers)

        # Register tools with the model if supported
        self._register_tools()

    #############
    # Callbacks #
    #############

    def _default_on_tool_use(
        self, tool_name: str, parameters: dict, result: Any
    ) -> None:
        """Default callback for tool use.

        :param tool_name: The name of the tool being called
        :param parameters: The parameters passed to the tool
        :param result: The result returned by the tool
        """
        if self.log_agentic_response:
            self.log(
                f"Using tool [{tool_name}] with parameters: {parameters} -> {result}",
                AgentLogType.TOOL_USE,
            )

    def _default_on_thinking(self, text: str) -> None:
        """Default callback for agent thinking.

        :param text: The text chunk produced by the agent
        """
        if self.log_agentic_response:
            self.log(text, AgentLogType.THOUGHT)

    def _default_on_answer(self, text: str) -> None:
        """Default callback for agent answers.

        :param text: The text chunk produced by the agent
        """
        if self.log_agentic_response:
            self.log(text, AgentLogType.ANSWER)

    def _set_event_handlers(
        self, event_handlers: EventHandler | dict[str, HandlerType] | None = None
    ) -> None:
        """Set up the event handlers. If `log_agentic_response` is True, include default handlers.

        :param event_handlers: Event handlers config for different agent events
        """

        # If we have event handlers that are passed as a dictionary, validate them
        if event_handlers is not None and not isinstance(event_handlers, EventHandler):
            try:
                event_handlers = EventHandler(**event_handlers)
            except Exception as e:
                raise ValueError(f"Invalid event handlers: {e}")

        # Start with empty set of handlers
        self.event_handlers = {}

        # Add default handlers if logging is enabled
        if self.log_agentic_response:
            self.event_handlers = {
                "on_tool_use": [self._default_on_tool_use],
                "on_thinking": [self._default_on_thinking],
                "on_answer": [self._default_on_answer],
            }

        # Process and merge user-provided handlers if any
        if event_handlers:
            # Convert to dict for processing
            handlers_dict = event_handlers.model_dump(exclude_none=True)
            for event_name, handler in handlers_dict.items():
                if event_name in self.event_handlers:
                    # Convert handler to list if it's not already
                    handlers_to_add = (
                        [handler] if not isinstance(handler, list) else handler
                    )
                    self.event_handlers[event_name].extend(handlers_to_add)
                else:
                    # For new event types, just use the provided handler(s)
                    self.event_handlers[event_name] = (
                        handler if isinstance(handler, list) else [handler]
                    )

    def _safe_event_handler(self, event_name: str, *args, **kwargs) -> None:
        """Safely dispatch one or more event handlers, handling cases where the event handler isn't assigned.

        :param event_name: The name of the event to invoke
        :param args: Positional arguments to pass to the event handler
        :param kwargs: Keyword arguments to pass to the event handler
        """
        handlers = self.event_handlers.get(event_name)
        if not handlers:
            return

        # Convert single handler to list for uniform processing
        if not isinstance(handlers, list):
            handlers = [handlers]

        # Execute each handler safely
        for handler in handlers:
            if not callable(handler):
                logger.warning(f"Event handler for {event_name} is not callable")
                continue

            try:
                handler(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Error in {event_name} event handler: {e}")
                # Continue execution even if event handler fails

    ###########
    # Helpers #
    ###########

    def _register_tools(self):
        """
        Register tools with the Bedrock model if supported.
        """

        # Check if the model has the toolConfig attribute
        if not hasattr(self.model, "toolConfig"):
            logger.warning(
                f"Model {self.model.__class__.__name__} does not support tool registration"
            )
            return

        # If no tools are provided, clear the toolConfig
        if not self.tools:
            self.model.toolConfig = None
            return

        # Convert BaseTool objects to BedrockTool format and collect them efficiently
        bedrock_tools = [
            BedrockTool(**tool.model_dump_bedrock_converse()) for tool in self.tools
        ]

        # Set the toolConfig on the model
        self.model.toolConfig = BedrockToolConfig(tools=bedrock_tools)
        logger.info(f"Registered {len(bedrock_tools)} tools with Bedrock model")

    def _process_tool_call(self, tool_name: str, tool_parameters: dict) -> str:
        """
        Process a single tool call.

        :param tool_name: Name of the tool to call
        :param tool_parameters: Parameters to pass to the tool
        :return: The formatted tool result message
        """
        tool = self._find_tool(tool_name)
        if not tool:
            return f"[Tool Error: {tool_name}] Tool not found"

        try:
            return self._execute_tool(tool, tool_parameters)
        except Exception as e:
            return self._handle_tool_error(tool_name, str(e))

    def _find_tool(self, tool_name: str) -> BaseTool | None:
        """Find a tool by name.

        :param tool_name: Name of the tool to find
        :return: The tool if found, None otherwise
        """
        for tool in self.tools:
            if tool.get_tool_name() == tool_name:
                return tool
        return None

    def _execute_tool(self, tool: BaseTool, parameters: dict) -> str:
        """Execute a tool with the given parameters.

        :param tool: Tool to execute
        :param parameters: Parameters to pass to the tool
        :return: The formatted tool result message
        """
        tool_result = tool.run(environment=self.environment, **parameters)

        # Call the observation callback
        self._safe_event_handler(
            event_name="on_tool_use",
            tool_name=tool.get_tool_name(),
            parameters=parameters,
            result=tool_result,
        )

        # Create tool message for logs
        tool_result_message = (
            f"[Tool Result] {tool.get_tool_name()}({parameters}) -> {tool_result}"
        )

        # Add tool result to memory as user message
        self.memory.add_message(
            role="user",
            content=f"[SYSTEM DIRECTIVE] The {tool.get_tool_name()} returned: {tool_result}. DO NOT ACKNOWLEDGE THIS MESSAGE. FIRST THINK, THEN PROCEED IMMEDIATELY to either: (1) Call the next required tool without any introduction or transition phrases, or (2) If all necessary tools have been used, provide your final answer. Think back to the original user prompt and use that to guide your response.",
        )

        return tool_result_message

    def _handle_tool_error(self, tool_name: str, error: str) -> str:
        """Handle a tool execution error.

        :param tool_name: Name of the tool that failed
        :param error: Error message
        :return: The formatted error message
        """
        error_msg = f"[Tool Error: {tool_name}] {error}"

        # Log the error using the observation callback
        self._safe_event_handler(
            event_name="on_tool_use",
            tool_name=tool_name,
            result=f"Error: {error}",
        )

        # Add error to memory
        self.memory.add_message(
            role="user",
            content=f"Your tool call resulted in an error: {error}. Please try a different approach.",
        )

        return error_msg

    def _process_tool_data(
        self, tool_data: dict, stream_handler: StreamHandler
    ) -> None:
        """Process collected tool data.

        :param tool_data: Tool data to process
        :param stream_handler: The stream handler to use for storing tool usages
        """
        if not tool_data.get("toolName") or not tool_data.get("parameters"):
            return

        # Find the tool by name
        tool = self._find_tool(tool_data["toolName"])
        raw_result = None

        if not tool:
            raw_result = f"Error: Tool {tool_data['toolName']} not found"

            # Call the observation callback with the error
            self._safe_event_handler(
                event_name="on_tool_use",
                tool_name=tool_data["toolName"],
                parameters=tool_data["parameters"],
                result=raw_result,
            )

            # Add error to memory
            self.memory.add_message(
                role="user",
                content=f"Tool '{tool_data['toolName']}' not found. Please try a different approach.",
            )
        else:
            try:
                # Execute the tool directly to get the raw result
                raw_result = tool.run(
                    environment=self.environment, **tool_data["parameters"]
                )

                # Call the observation callback with the raw result
                self._safe_event_handler(
                    event_name="on_tool_use",
                    tool_name=tool_data["toolName"],
                    parameters=tool_data["parameters"],
                    result=raw_result,
                )

                # Add tool result to memory as user message
                self.memory.add_message(
                    role="user",
                    content=f"[SYSTEM DIRECTIVE] The {tool_data['toolName']} returned: {raw_result}. DO NOT ACKNOWLEDGE THIS MESSAGE. FIRST THINK, THEN PROCEED IMMEDIATELY to either: (1) Call the next required tool without any introduction or transition phrases, or (2) If all necessary tools have been used, provide your final answer. Think back to the original user prompt and use that to guide your response.",
                )
            except Exception as e:
                error_msg = str(e)
                raw_result = f"Error: {error_msg}"

                # Log the error using the observation callback
                self._safe_event_handler(
                    event_name="on_tool_use",
                    tool_name=tool_data["toolName"],
                    parameters=tool_data["parameters"],
                    result=raw_result,
                )

                # Add error to memory
                self.memory.add_message(
                    role="user",
                    content=f"Your tool call resulted in an error: {error_msg}. Please try a different approach.",
                )

        # Store tool usage as an event with the raw result
        stream_handler.events.append(
            {
                "type": "tool_usage",
                "content": {
                    "name": tool_data["toolName"],
                    "parameters": tool_data["parameters"],
                    "result": raw_result,  # Store the raw result
                },
            }
        )

    def _process_content(self, content: str) -> tuple[list[str], list[str]]:
        """
        Process the content to extract thinking and answer parts.

        :param content: The content to process
        :return: A tuple of (thinking, answer)
        """

        # Find all thinking and answer tags
        thoughts = [match.strip() for match in self._THINKING_PATTERN.findall(content)]
        answers = [match.strip() for match in self._ANSWER_PATTERN.findall(content)]

        # Trigger callbacks for thinking and answer
        for thought in thoughts:
            self._safe_event_handler(event_name="on_thinking", text=thought)
        for answer in answers:
            self._safe_event_handler(event_name="on_answer", text=answer)

        # If no matches are found, return empty lists
        if not thoughts and not answers:
            return [], []

        return thoughts, answers

    ##########
    # Invoke #
    ##########

    def _invoke_iteration(self, prompt_obj: Messages) -> dict[str, Any]:
        """
        Process a single iteration of the agent's conversation with the model using invoke.

        :param prompt_obj: Messages object containing the conversation history
        :return: Dictionary with response data including content, thinking, answer, and tool data
        """
        # Get the full response from the model using invoke
        response = self.model.invoke(prompt=prompt_obj)

        # Debug log the response structure
        logger.debug(f"Response structure: {response}")

        # Extract content depending on response structure
        content = ""
        tool_used = False
        tool_name = None
        tool_parameters = {}
        tool_results = []
        tool_data = None

        # Process Bedrock response structure
        if isinstance(response, dict):

            # Get the message from the response
            if "output" in response and "message" in response["output"]:
                message = response["output"]["message"]

                # Check if content is an array of items
                if "content" in message and isinstance(message["content"], list):
                    for item in message["content"]:
                        if isinstance(item, dict):
                            # Extract text content from text blocks
                            if "text" in item:
                                content += item["text"]

                            # Extract tool use information
                            if "toolUse" in item:
                                tool_used = True
                                tool_info = item["toolUse"]
                                tool_name = tool_info.get("name")

                                # Extract parameters from input
                                if "input" in tool_info and isinstance(
                                    tool_info["input"], dict
                                ):
                                    tool_parameters = tool_info["input"]

                # Check stopReason for tool_use to confirm tool usage
                if "stopReason" in response and response["stopReason"] == "tool_use":
                    tool_used = True
            else:
                # For logging purposes only
                logger.warning("Couldn't find content in response structure")
        elif isinstance(response, str):
            # Direct string response
            content = response

        # Handle content
        if content:
            # Process the content to handle thinking/answer tags
            thinking, answer = self._process_content(content)

            # Add assistant's response to memory
            self.memory.add_message(role="assistant", content=content)

        # Process tool call if found in the response
        if tool_used and tool_name and tool_parameters:
            # Find the tool and execute it
            tool = self._find_tool(tool_name)
            if tool:
                try:
                    # Execute the tool directly to get the raw result
                    tool_result = tool.run(
                        environment=self.environment, **tool_parameters
                    )

                    # Create formatted result message for logs
                    formatted_result = (
                        f"[Tool Result] {tool_name}({tool_parameters}) -> {tool_result}"
                    )
                    tool_results.append(formatted_result)

                    # Store structured tool data
                    tool_data = {
                        "name": tool_name,
                        "parameters": tool_parameters,
                        "result": tool_result,
                    }

                    # Call the observation callback
                    self._safe_event_handler(
                        event_name="on_tool_use",
                        tool_name=tool_name,
                        parameters=tool_parameters,
                        result=tool_result,
                    )

                    # Add tool result to memory
                    self.memory.add_message(
                        role="user",
                        content=f"[SYSTEM DIRECTIVE] The {tool_name} returned: {tool_result}. DO NOT ACKNOWLEDGE THIS MESSAGE. FIRST THINK, THEN PROCEED IMMEDIATELY to either: (1) Call the next required tool without any introduction or transition phrases, or (2) If all necessary tools have been used, provide your final answer. Think back to the original user prompt and use that to guide your response.",
                    )
                except Exception as e:
                    error_msg = str(e)
                    formatted_result = f"[Tool Error: {tool_name}] {error_msg}"
                    tool_results.append(formatted_result)
                    tool_data = {
                        "name": tool_name,
                        "parameters": tool_parameters,
                        "result": f"Error: {error_msg}",
                    }

                    # Log the error
                    self._safe_event_handler(
                        event_name="on_tool_use",
                        tool_name=tool_name,
                        parameters=tool_parameters,
                        result=f"Error: {error_msg}",
                    )

                    # Add error to memory
                    self.memory.add_message(
                        role="user",
                        content=f"Your tool call resulted in an error: {error_msg}. Please try a different approach.",
                    )
            else:
                # Tool not found
                formatted_result = f"[Tool Error: {tool_name}] Tool not found"
                tool_results.append(formatted_result)
                tool_data = {
                    "name": tool_name,
                    "parameters": tool_parameters,
                    "result": "Error: Tool not found",
                }

                # Log the error
                self._safe_event_handler(
                    event_name="on_tool_use",
                    tool_name=tool_name,
                    result="Error: Tool not found",
                )

                # Add error to memory
                self.memory.add_message(
                    role="user",
                    content=f"Tool '{tool_name}' not found. Please try a different approach.",
                )

            return {
                "content": content,
                "thinking": thinking,
                "answer": answer,
                "tool_used": True,
                "tool_data": tool_data,
                "tool_results": tool_results,
            }

        return {
            "content": content,
            "thinking": thinking,
            "answer": answer,
            "tool_used": False,
            "tool_data": None,
            "tool_results": tool_results,
        }

    def invoke(self, prompt: str, max_iterations: int = 10) -> dict[str, Any]:
        """
        Run the agent with the given prompt using the model's invoke method.

        :param prompt: The initial prompt to feed to the LLM
        :param max_iterations: Maximum number of model-tool-model iterations
        :return: A dictionary containing 'content', 'thinking', 'tool_use', and 'answer'
        """
        # Clear or reset the agent's memory context
        self._flush_memory()

        # Add the user's prompt to memory
        self.memory.add_message(role="user", content=prompt)

        # Initialize buffers for thinking and answer
        all_thinking = []
        all_answer = []
        all_tool_use = []
        full_response = []
        iterations = 0

        while iterations < max_iterations:

            # Get current messages and system
            messages = self.memory.get_messages()
            system = self.memory.get_system_message()

            # Prep messages for the model
            prompt_obj = Messages(system=system, messages=messages)

            # Process one iteration and get response and tool use flag
            response = self._invoke_iteration(prompt_obj)

            # Add thoughts and answer to the buffers
            all_thinking.extend(response["thinking"])
            if response["answer"] is not None:
                if isinstance(response["answer"], list):
                    all_answer.extend(response["answer"])
                else:
                    all_answer.append(response["answer"])

            # Add structured tool data if a tool was used
            if response["tool_used"] and response["tool_data"]:
                all_tool_use.append(response["tool_data"])

            # Add to the full response if we got a valid response
            if response:
                full_response.append(response["content"])

            # Add tool result messages to the full response
            if response["tool_results"]:
                full_response.extend(response["tool_results"])

            # If no tool was used, we're done
            if not response["tool_used"]:
                break

            # Increment iteration counter
            iterations += 1

            # If we've hit max iterations, exit
            if iterations >= max_iterations:
                logger.warning(
                    f"Reached maximum iterations ({max_iterations}). Stopping."
                )
                break

        # Get the final answer if any was extracted
        answer = all_answer[0] if all_answer else None

        return {
            "content": "\n\n".join(full_response),
            "thinking": all_thinking,
            "tool_use": all_tool_use,
            "answer": answer,
        }

    ###########
    # Helpers #
    ###########

    def _handle_text_delta(
        self, delta: dict, stream_handler: StreamHandler, answer_content: str | None
    ) -> Generator[str, None, str | None]:
        """Handle a text delta from the stream.

        :param delta: The delta to process
        :param stream_handler: The stream handler to use
        :param answer_content: Current answer content
        :return: Updated answer content
        """
        if "text" in delta:
            text = delta["text"]
            stream_handler.process_chunk(text)
            # If we have answer content, yield it and reset
            if answer_content is not None:
                yield answer_content
                answer_content = None
        return answer_content

    def _handle_tool_delta(
        self, delta: dict, is_collecting_tool_data: bool, current_tool_buffer: str
    ) -> tuple[bool, str]:
        """Handle a tool delta from the stream.

        :param delta: The delta to process
        :param is_collecting_tool_data: Whether we're currently collecting tool data
        :param current_tool_buffer: Current tool buffer
        :return: Tuple of (is_collecting_tool_data, current_tool_buffer)
        """
        if (
            "toolUse" in delta
            and "input" in delta["toolUse"]
            and is_collecting_tool_data
        ):
            # Accumulate the input string before trying to parse it
            current_tool_buffer += delta["toolUse"].get("input", "")
        return is_collecting_tool_data, current_tool_buffer

    def _handle_tool_start(self, chunk: dict) -> tuple[bool, str, dict, str]:
        """Handle the start of a tool call.

        :param chunk: The chunk containing the tool start
        :return: Tuple of (is_collecting_tool_data, current_tool_name, current_tool_data, current_tool_buffer)
        """
        is_collecting_tool_data = True
        tool_info = chunk["contentBlockStart"]["start"]["toolUse"]
        current_tool_name = tool_info.get("name")
        current_tool_data = {"toolName": current_tool_name, "parameters": {}}
        current_tool_buffer = ""

        # Capture initial arguments if available in the start block
        if "arguments" in tool_info:
            try:
                arguments = tool_info.get("arguments", {})
                if isinstance(arguments, dict):
                    current_tool_data["parameters"].update(arguments)
                elif isinstance(arguments, str):
                    parsed_args = json.loads(arguments)
                    if isinstance(parsed_args, dict):
                        current_tool_data["parameters"].update(parsed_args)
            except Exception as e:
                logger.warning(f"Error parsing initial tool arguments: {e}")

        return (
            is_collecting_tool_data,
            current_tool_name,
            current_tool_data,
            current_tool_buffer,
        )

    def _handle_tool_stop(
        self,
        _chunk: dict,
        is_collecting_tool_data: bool,
        current_tool_name: str | None,
        current_tool_data: dict,
        current_tool_buffer: str,
        stream_handler: StreamHandler,
    ) -> tuple[bool, str | None, dict, str, bool]:
        """Handle the end of a tool call.

        :param _chunk: The chunk containing the tool stop (unused)
        :param is_collecting_tool_data: Whether we're currently collecting tool data
        :param current_tool_name: Current tool name
        :param current_tool_data: Current tool data
        :param current_tool_buffer: Current tool buffer
        :param stream_handler: The stream handler to use for storing tool usages
        :return: Tuple of (is_collecting_tool_data, current_tool_name, current_tool_data, current_tool_buffer, tool_used)
        """
        tool_used = False
        if current_tool_buffer:
            try:
                parsed_input = json.loads(current_tool_buffer)
                if isinstance(parsed_input, dict):
                    current_tool_data["parameters"].update(parsed_input)
            except Exception as e:
                logger.warning(
                    f"Error parsing tool input: {e}, input buffer: {current_tool_buffer}"
                )

        # Execute the tool if we have valid data
        if current_tool_name and current_tool_data:
            tool_used = True
            self._process_tool_data(current_tool_data, stream_handler)

        # Reset tool collection state
        is_collecting_tool_data = False
        current_tool_name = None
        current_tool_data = {}
        current_tool_buffer = ""

        return (
            is_collecting_tool_data,
            current_tool_name,
            current_tool_data,
            current_tool_buffer,
            tool_used,
        )

    def _handle_message_stop(
        self,
        chunk: dict,
        iteration_response: str,
        tool_used: bool,
        answer_content: str | None,
    ) -> Generator[str, None, tuple[str, str | None]]:
        """Handle the end of a message.

        :param chunk: The chunk containing the message stop
        :param iteration_response: Current iteration response
        :param tool_used: Whether a tool was used
        :param answer_content: Current answer content
        :return: Generator yielding answer content, finally returning (iteration_response, answer_content)
        """
        # Message is complete, add the final response to memory if it wasn't a tool call
        if iteration_response and not tool_used:
            self.memory.add_message(role="assistant", content=iteration_response)

        # Check if we need to report any metadata
        if "metadata" in chunk:
            logger.info(f"Response metadata: {chunk['metadata']}")

        # Yield any remaining answer content
        if answer_content is not None:
            yield answer_content
            answer_content = None

        return iteration_response, answer_content

    def _unpack_event_stream(self, events: list[dict]) -> dict[str, Any]:
        """Unpack the event list into a dictionary with standardized format.

        Transforms a list of events into a dictionary with keys:
        - content: Full combined response text
        - thinking: List of all thinking entries
        - tool_use: List of dicts with tool name, parameters, and result
        - answer: Final answer text (combined)

        :param events: The events to unpack
        :return: Dictionary with standardized output format
        """
        # Initialize result containers
        thinking = []
        answer_chunks = []
        tool_use = []
        full_content = []

        # Process each event
        for event in events:
            event_type = event["type"]

            if event_type == "thinking":
                thinking.append(event["content"])
                full_content.append(f"<thinking>{event['content']}</thinking>")

            elif event_type == "answer":
                answer_chunks.append(event["content"])
                full_content.append(f"<answer>{event['content']}</answer>")

            elif event_type == "tool_usage":
                # Extract tool data and add to tool_use list as a dict
                tool_data = event["content"]
                tool_use.append(
                    {
                        "name": tool_data["name"],
                        "parameters": tool_data["parameters"],
                        "result": tool_data["result"],
                    }
                )

                # Add formatted tool usage to full content
                tool_message = f"[Tool Result] {tool_data['name']}({tool_data['parameters']}) -> {tool_data['result']}"
                full_content.append(tool_message)

        # Combine answer chunks into a single answer
        answer = "".join(answer_chunks) if answer_chunks else None

        # Return a dictionary with the same structure as invoke() method
        return {
            "content": "\n\n".join(full_content),
            "thinking": thinking,
            "tool_use": tool_use,
            "answer": answer,
            "events": events,
        }

    ##########
    # Stream #
    ##########

    def _stream_iteration(self, prompt_obj: Messages) -> Iterator[str]:
        """
        Stream a single iteration of the agent's conversation with the model.

        :param prompt_obj: Messages object containing the conversation history
        :return: Iterator of response chunks
        """
        # Track the response for this iteration
        iteration_response = ""
        tool_used = False

        # State tracking for tool use - initialize only what's needed
        current_tool_data = {}
        is_collecting_tool_data = False
        current_tool_name = None
        current_tool_buffer = ""

        # Create a stream handler for processing tagged content
        def handle_tagged_content(tag_name: str, content: str) -> None:
            """Handle content from tagged sections.

            :param tag_name: Name of the tag (thinking, answer, etc)
            :param content: Content within the tag
            """
            nonlocal iteration_response

            # Add to iteration response
            iteration_response += f"<{tag_name}>{content}</{tag_name}>"

            # Call appropriate event handler
            if tag_name == "thinking":
                self._safe_event_handler(event_name="on_thinking", text=content)
            elif tag_name == "answer":
                self._safe_event_handler(event_name="on_answer", text=content)
                # Store the content to be yielded later
                nonlocal answer_content
                answer_content = content

        # Initialize answer_content to be used in the generator
        answer_content = None
        stream_handler = StreamHandler(event_callback=handle_tagged_content)

        # Stream and process the response
        for chunk in self.model.stream(prompt=prompt_obj):
            # Handle different chunk types
            if "contentBlockDelta" in chunk:
                delta = chunk["contentBlockDelta"]["delta"]

                # Handle text deltas
                if "text" in delta:
                    text = delta["text"]
                    stream_handler.process_chunk(text)
                    # If we have answer content, yield it and reset
                    if answer_content is not None:
                        yield answer_content
                        answer_content = None

                # Handle tool use input collection
                elif (
                    "toolUse" in delta
                    and "input" in delta["toolUse"]
                    and is_collecting_tool_data
                ):
                    # Accumulate the input string before trying to parse it
                    current_tool_buffer += delta["toolUse"].get("input", "")

            # Handle tool call start
            elif "contentBlockStart" in chunk and "toolUse" in chunk.get(
                "contentBlockStart", {}
            ).get("start", {}):
                (
                    is_collecting_tool_data,
                    current_tool_name,
                    current_tool_data,
                    current_tool_buffer,
                ) = self._handle_tool_start(chunk)

            # Handle end of tool call collection
            elif "contentBlockStop" in chunk and is_collecting_tool_data:
                # Tool data collection is complete, parse the accumulated input buffer
                if current_tool_buffer:
                    try:
                        parsed_input = json.loads(current_tool_buffer)
                        if isinstance(parsed_input, dict):
                            current_tool_data["parameters"].update(parsed_input)
                    except Exception as e:
                        logger.warning(
                            f"Error parsing tool input: {e}, input buffer: {current_tool_buffer}"
                        )

                # Execute the tool if we have valid data
                if current_tool_name and current_tool_data:
                    tool_used = True
                    # Reset tool collection state
                    (
                        is_collecting_tool_data,
                        current_tool_name,
                        current_tool_data,
                        current_tool_buffer,
                        tool_used,
                    ) = self._handle_tool_stop(
                        chunk,
                        is_collecting_tool_data,
                        current_tool_name,
                        current_tool_data,
                        current_tool_buffer,
                        stream_handler,
                    )

            # Handle message end
            elif "messageStop" in chunk:
                # Message is complete, add the final response to memory if it wasn't a tool call
                if iteration_response and not tool_used:
                    self.memory.add_message(
                        role="assistant", content=iteration_response
                    )

                # Check if we need to report any metadata
                if "metadata" in chunk:
                    logger.info(f"Response metadata: {chunk['metadata']}")

                # Yield any remaining answer content
                if answer_content is not None:
                    yield answer_content
                    answer_content = None

        # Return tool_used as a hidden value for the stream method
        if tool_used:
            # This is a special marker that will not be displayed but will be detected by the stream method
            yield "[[TOOL_USED]]"

        # Return the collected data
        yield {
            "events": stream_handler.events,
        }

    def stream(self, prompt: str, max_iterations: int = 10) -> Iterator[str]:
        """
        Stream the agent's response with the given prompt.

        :param prompt: The initial prompt to feed to the LLM
        :param max_iterations: Maximum number of model-tool-model iterations
        :return: Iterator of response chunks
        """
        # Clear or reset the agent's memory context
        self._flush_memory()

        # Add the user's prompt to memory
        self.memory.add_message(role="user", content=prompt)

        iterations = 0
        all_events = []

        while iterations < max_iterations:
            iterations += 1

            # Get current messages and system
            messages = self.memory.get_messages()
            system = self.memory.get_system_message()

            # Prep messages for the model
            prompt_obj = Messages(system=system, messages=messages)

            # Stream and process the response, tracking if a tool was used
            tool_used = False

            for chunk in self._stream_iteration(prompt_obj):
                # Check if this is our special marker for tool usage
                if chunk == "[[TOOL_USED]]":
                    tool_used = True
                    # Don't yield this special marker
                    continue

                # Check if this is our data collection chunk
                if isinstance(chunk, dict):
                    # Collect the events
                    all_events.extend(chunk["events"])
                    # Don't yield the data collection chunk
                    continue

                # Yield all other chunks normally
                yield chunk

            # If no tool was used in this iteration, we're done
            if not tool_used:
                break

            # If we've hit max iterations, exit
            if iterations >= max_iterations:
                logger.warning(
                    f"Reached maximum iterations ({max_iterations}). Stopping."
                )
                break

        # Yield the final collected data
        yield {
            "events": all_events,
        }

    def run(
        self, prompt: str, max_iterations: int = 10, stream: bool = False
    ) -> dict[str, Any] | Iterator[str] | dict[str, list]:
        """
        Run the agent with the given prompt.

        :param prompt: The initial prompt to feed to the LLM
        :param max_iterations: Maximum number of model-tool-model iterations
        :param stream: Whether to stream the response or return the full text
        :return: If stream=False: A dictionary containing 'content', 'thinking', 'tool_use', and 'answer'
                If stream=True: A dictionary containing 'events' in chronological order
        """
        # If streaming is requested, use the stream method directly
        if stream:
            # Create collection for all events
            all_events = []

            # Process the stream and collect data
            for chunk in self.stream(prompt, max_iterations):
                # Check if this is our data collection chunk
                if isinstance(chunk, dict):
                    # Collect the events
                    all_events.extend(chunk["events"])

            # Return the unpacked events in the same format as invoke()
            return self._unpack_event_stream(all_events)

        response = self.invoke(prompt, max_iterations)
        return response


if __name__ == "__main__":
    import json

    from fence.models.bedrock.claude import Claude35Sonnet
    from fence.tools.base import BaseTool, tool
    from fence.utils.logger import setup_logging

    setup_logging(log_level="INFO", are_you_serious=False)

    # Define a weather tool using the decorator
    @tool(description="Get the current weather for a location")
    def get_weather(location: str):
        """
        Get the current weather for a location.

        :param location: The location to get the weather for
        :return: The current weather for the location
        """
        # Simulate weather API response
        weather_data = {
            "New York": "Sunny, 75°F",
            "London": "Rainy, 55°F",
            "Tokyo": "Cloudy, 65°F",
            "Sydney": "Clear, 80°F",
        }
        return weather_data.get(location, f"Weather data not available for {location}")

    # Define a temperature conversion tool
    @tool(description="Convert temperature between Fahrenheit and Celsius")
    def convert_temperature(value: float, from_unit: str, to_unit: str):
        """
        Convert temperature between Fahrenheit and Celsius.

        :param value: The temperature value to convert
        :param from_unit: The unit to convert from ('Fahrenheit', 'F', 'Celsius', 'C')
        :param to_unit: The unit to convert to ('Fahrenheit', 'F', 'Celsius', 'C')
        :return: The converted temperature value
        """
        # Normalize input units
        from_unit = from_unit.lower()
        to_unit = to_unit.lower()

        # Convert full names to single letters
        if from_unit in ["fahrenheit", "f"]:
            from_unit = "f"
        elif from_unit in ["celsius", "c"]:
            from_unit = "c"

        if to_unit in ["fahrenheit", "f"]:
            to_unit = "f"
        elif to_unit in ["celsius", "c"]:
            to_unit = "c"

        if from_unit == "f" and to_unit == "c":
            return f"{(value - 32) * 5/9:.1f}°C"
        elif from_unit == "c" and to_unit == "f":
            return f"{(value * 9/5) + 32:.1f}°F"
        else:
            return f"Invalid conversion: {from_unit} to {to_unit}"

    # Create a prompt that requires multiple tool calls
    prompt = "What's the value of the USD in EUR, today?"

    #################
    # Non-streaming #
    #################

    logger.info("\nExample 1: Using BedrockAgent with multiple tools:")
    setup_logging(log_level="DEBUG", are_you_serious=False)

    # Define a event handler class to avoid using globals
    class DummyEventHandler:
        """Event handler for managing agent event handlers with internal state."""

        def on_tool_use(self, tool_name, parameters, result):
            """Handle tool use events."""
            print(f"🔧 TOOL: {tool_name} with {parameters} -> {result}")

        def on_thinking(self, text):
            """Handle agent thinking events."""
            print(f"🧠 THINKING: {text}")

        def on_answer(self, text):
            print(f"💬 ANSWER: {text}")

    # Create the event handler instance
    event_handler = DummyEventHandler()

    # Example 3: Using BedrockAgent with multiple tools
    agent = BedrockAgent(
        identifier="WeatherAssistant",
        model=Claude35Sonnet(),
        description="An assistant that can provide weather information and perform temperature conversions",
        # tools=[get_weather, convert_temperature],
        memory=FleetingMemory(),
        log_agentic_response=True,  # Disable default logging since we're using callbacks
        system_message="Be very courteous.",
        event_handlers={
            "on_tool_use": event_handler.on_tool_use,
            "on_thinking": event_handler.on_thinking,
            "on_answer": event_handler.on_answer,
        },
    )

    response = agent.run(prompt, stream=False)

    from pprint import pprint

    pprint(response)

    #############
    # Streaming #
    #############

    # logger.info("\nExample 2: Using BedrockAgent with streaming:")

    # # Create a new agent with streaming enabled
    # agent = BedrockAgent(
    #     identifier="WeatherAssistant",
    #     model=NovaPro(region="us-east-1"),
    #     description="An assistant that can provide weather information and perform temperature conversions",
    #     tools=[get_weather, convert_temperature],
    #     memory=FleetingMemory(),
    #     log_agentic_response=True,  # Disable default logging since we're using callbacks
    #     system_message="Answer like a pirate",
    # )

    # # Use streaming mode
    # collected_data = agent.run(prompt, stream=True)

    # # Access the answer
    # print(f"Answer: {collected_data['answer']}")

    #########################
    # Prompt model directly #
    #########################

    # Create a custom event handler
    import random

    from fence.agents.bedrock import EventHandler

    def on_tool_use(tool_name, parameters, result):
        """Handle tool use events."""
        print(
            f"SENDING TOOL USE TO SLACK: CALLED A TOOL: {tool_name} with {parameters} -> {result}"
        )

    def on_thinking(text):
        """Handle agent thinking events."""
        synonyms_for_thinking = [
            "thinking",
            "pondering",
            "considering",
            "evaluating",
            "analyzing",
            "reflecting",
            "considering",
            "evaluating",
            "analyzing",
            "reflecting",
            "considering",
            "evaluating",
            "analyzing",
            "reflecting",
        ]
        print(f"SENDING THINKING TO SLACK: *{random.choice(synonyms_for_thinking)}*")

    def on_answer(text):
        """Handle agent answer events."""
        print(f"SENDING ANSWER TO SLACK: {text}")

    event_handler = EventHandler(
        on_tool_use=on_tool_use, on_thinking=on_thinking, on_answer=on_answer
    )

    # Create the agent
    agent = BedrockAgent(model=Claude35Sonnet(), event_handlers=event_handler)

    # Run the agent
    agent.run("What is the weather in New York, in Celsius?")
