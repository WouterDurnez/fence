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
    :param on_delegate: Called when the agent delegates to another agent
    """

    on_tool_use: HandlerType | None = None
    on_thinking: HandlerType | None = None
    on_answer: HandlerType | None = None
    on_delegate: HandlerType | None = None

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
You are a helpful assistant. You can think in <thinking> tags, and provide an answer in <answer> tags. Try to always plan your next steps in <thinking> tags.
"""

    _THINKING_PATTERN = re.compile(r"<thinking>(.*?)</thinking>", re.DOTALL)
    _ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
    _DELEGATE_PATTERN = re.compile(r"<delegate>(.*?)</delegate>", re.DOTALL)

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
        delegates: list[BaseAgent] | None = None,
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
        :param delegates: A list of delegate agents available to the agent
        :param system_message: A system message to set for the agent
        :param event_handlers: Event handlers for different agent events (on_tool_use, on_thinking, on_answer, on_delegate)
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
        self._system_message = self._BASE_SYSTEM_MESSAGE

        # Store the user-provided system message for later rebuilds
        self._user_system_message = system_message

        # Set the tools
        self.tools = tools or []

        # Set the delegates as a list
        self.delegates = self._setup_delegates(delegates or [])

        # Update with user-provided event handlers
        self._set_event_handlers(event_handlers)

        # Add delegation options to system message if delegates are available
        self._system_message = self._build_system_message(system_message)

        # Set the system message in memory
        self.memory.set_system_message(self._system_message)

        # Register tools with the model if supported
        self._register_tools()

    def _build_system_message(self, user_system_message: str | None = None) -> str:
        """Build the system message with delegation options based on available delegates.

        :param user_system_message: Optional user-provided system message to append
        :return: Complete system message including delegation options
        """
        system_message = self._BASE_SYSTEM_MESSAGE

        # Add delegation instructions if delegates are available
        if self.delegates:
            delegate_info = "\nYou can also delegate to the following agents using <delegate>agent_name:query</delegate> tags:"

            # Add information about each delegate
            for delegate in self.delegates:
                delegate_name = delegate.identifier
                delegate_desc = delegate.description or "No description available"
                delegate_info += f"\n- {delegate_name}: {delegate_desc}"

            system_message += delegate_info

        # Use the provided user_system_message or fall back to stored one
        effective_user_message = (
            user_system_message
            if user_system_message is not None
            else self._user_system_message
        )

        # Append user system message if available
        if effective_user_message:
            system_message += f"\n\n{effective_user_message}"

        return system_message

    def _setup_delegates(self, delegates: list[BaseAgent]) -> list[BaseAgent]:
        """Set up delegate agents with environment variables.

        :param delegates: List of delegate agents
        :return: List of properly set up delegate agents
        """
        valid_delegates = []
        for delegate in delegates:
            identifier = delegate.identifier
            if not identifier:
                logger.warning(
                    f"Delegate without identifier will be skipped: {delegate}"
                )
                continue

            valid_delegates.append(delegate)
            # Pass environment variables to delegate
            delegate.environment.update(self.environment)

        return valid_delegates

    def update_delegates(self, delegates: list[BaseAgent]) -> None:
        """Update the agent's delegates and refresh the system message.

        :param delegates: New list of delegate agents to use
        """
        self.delegates = self._setup_delegates(delegates)

        # Rebuild and update the system message with new delegate information
        self._system_message = self._build_system_message()
        self.memory.set_system_message(self._system_message)

        delegate_names = [delegate.identifier for delegate in self.delegates]
        logger.info(f"Updated delegates: {delegate_names}")

    def add_delegate(self, delegate: BaseAgent) -> None:
        """Add a single delegate to the agent and update the system message.

        :param delegate: Delegate agent to add
        """
        if not delegate.identifier:
            logger.warning("Cannot add delegate without identifier")
            return

        # Skip if delegate with same identifier already exists
        if any(d.identifier == delegate.identifier for d in self.delegates):
            logger.warning(
                f"Delegate with identifier {delegate.identifier} already exists"
            )
            return

        # Pass environment variables to delegate
        delegate.environment.update(self.environment)

        # Add the delegate to our collection
        self.delegates.append(delegate)

        # Rebuild and update the system message
        self._system_message = self._build_system_message()
        self.memory.set_system_message(self._system_message)

        logger.info(f"Added delegate: {delegate.identifier}")

    def remove_delegate(self, identifier: str) -> bool:
        """Remove a delegate by identifier and update the system message.

        :param identifier: Identifier of the delegate to remove
        :return: True if delegate was removed, False if not found
        """
        for i, delegate in enumerate(self.delegates):
            if delegate.identifier == identifier:
                self.delegates.pop(i)

                # Rebuild and update the system message
                self._system_message = self._build_system_message()
                self.memory.set_system_message(self._system_message)

                logger.info(f"Removed delegate: {identifier}")
                return True

        logger.warning(f"Delegate not found: {identifier}")
        return False

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

    def _default_on_delegate(
        self, delegate_name: str, query: str, result: str | None
    ) -> None:
        """Default callback for delegate use.

        :param delegate_name: The name of the delegate agent
        :param query: The query passed to the delegate
        :param result: The result returned by the delegate (None if before execution)
        """
        if self.log_agentic_response:
            if result is None:
                # Delegation initiation
                self.log(
                    f'Initiating delegation to {delegate_name} with query: "{query}"',
                    AgentLogType.DELEGATE,
                )
            else:
                # Delegation completion - include delegate name but focus on result
                self.log(
                    f"Delegation to {delegate_name} concluded: {result}",
                    AgentLogType.DELEGATE,
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
                "on_delegate": [self._default_on_delegate],
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

        For delegation events, this method can handle both the initiation phase (when result=None)
        and the completion phase (when result contains the actual response).

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

        # Store event in chronological order for tracking
        if hasattr(self, "_current_iteration_events"):
            # For thinking events
            if event_name == "on_thinking" and "text" in kwargs:
                self._current_iteration_events.append(
                    {"type": "thinking", "content": kwargs["text"]}
                )
            # For answer events
            elif event_name == "on_answer" and "text" in kwargs:
                self._current_iteration_events.append(
                    {"type": "answer", "content": kwargs["text"]}
                )
            # For delegate events
            elif (
                event_name == "on_delegate"
                and "delegate_name" in kwargs
                and "query" in kwargs
            ):
                delegate_event = {
                    "type": "delegate_usage",
                    "content": {
                        "agent_name": kwargs["delegate_name"],
                        "query": kwargs["query"],
                    },
                }
                # Include result if available (completion phase)
                if "result" in kwargs and kwargs["result"] is not None:
                    delegate_event["content"]["result"] = kwargs["result"]
                self._current_iteration_events.append(delegate_event)
            # For tool events
            elif event_name == "on_tool_use" and "tool_name" in kwargs:
                tool_event = {
                    "type": "tool_usage",
                    "content": {
                        "name": kwargs["tool_name"],
                        "parameters": kwargs.get("parameters", {}),
                    },
                }
                # Include result if available
                if "result" in kwargs:
                    tool_event["content"]["result"] = kwargs["result"]
                self._current_iteration_events.append(tool_event)

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

    def _execute_delegate(self, delegate_name: str, query: str) -> str:
        """Execute a delegate agent with the given query.

        :param delegate_name: Name of the delegate agent to call
        :param query: Query to pass to the delegate
        :return: The result from the delegate agent
        """
        # Find the delegate by name
        delegate = next(
            (d for d in self.delegates if d.identifier == delegate_name), None
        )

        if not delegate:
            error_msg = f"Delegate agent '{delegate_name}' not found"
            logger.warning(error_msg)
            return error_msg

        try:
            # Call the delegation event handler BEFORE executing the delegate
            self._safe_event_handler(
                event_name="on_delegate",
                delegate_name=delegate_name,
                query=query,
                result=None,  # Result not available yet
            )

            # Call the delegate agent
            delegate_result = delegate.run(query)

            # Extract just the answer if it's a dict with multiple fields
            if isinstance(delegate_result, dict) and "answer" in delegate_result:
                result = delegate_result["answer"]
            else:
                result = str(delegate_result)

            # Update the event handler with the result
            self._safe_event_handler(
                event_name="on_delegate",
                delegate_name=delegate_name,
                query=query,
                result=result,
            )

            # Add delegate result to memory as user message
            self.memory.add_message(
                role="user",
                content=f"[SYSTEM DIRECTIVE] Delegated to {delegate_name} with query: {query}. Result: {result}. DO NOT ACKNOWLEDGE THIS MESSAGE. FIRST THINK, THEN PROCEED IMMEDIATELY to either: (1) Call the next required tool or delegate, or (2) If all necessary operations have been completed, provide your final answer. Think back to the original user prompt and use that to guide your response.",
            )

            return result
        except Exception as e:
            error_msg = f"Error executing delegate '{delegate_name}': {str(e)}"
            logger.warning(error_msg)

            # Call the observation callback with the error
            self._safe_event_handler(
                event_name="on_delegate",
                delegate_name=delegate_name,
                query=query,
                result=f"Error: {str(e)}",
            )

            # Add error to memory
            self.memory.add_message(
                role="user",
                content=f"Delegation to '{delegate_name}' failed with error: {str(e)}. Please try a different approach.",
            )

            return error_msg

    def _parse_delegate_tag(self, delegate_content: str) -> tuple[str, str]:
        """Parse the delegate tag content to extract agent name and query.

        :param delegate_content: Content of the delegate tag
        :return: Tuple of (agent_name, query)
        """
        # Expected format is "agent_name:query"
        parts = delegate_content.split(":", 1)
        if len(parts) != 2:
            return None, delegate_content

        agent_name = parts[0].strip()
        query = parts[1].strip()
        return agent_name, query

    def _process_content(
        self, content: str
    ) -> tuple[list[str], list[str], list[tuple[str, str, str]]]:
        """
        Process the content to extract thinking, answer, and delegate parts in the order they appear.

        :param content: The content to process
        :return: A tuple of (thinking, answer, delegates)
        """
        # Initialize return values
        thoughts = []
        answers = []
        delegates = []

        # Process all tags in chronological order
        remaining_content = content
        while remaining_content:
            # Find positions of the next tag of each type
            thinking_match = self._THINKING_PATTERN.search(remaining_content)
            answer_match = self._ANSWER_PATTERN.search(remaining_content)
            delegate_match = self._DELEGATE_PATTERN.search(remaining_content)

            thinking_pos = thinking_match.start() if thinking_match else float("inf")
            answer_pos = answer_match.start() if answer_match else float("inf")
            delegate_pos = delegate_match.start() if delegate_match else float("inf")

            # Find the earliest tag
            next_tag_type = None
            if thinking_pos < answer_pos and thinking_pos < delegate_pos:
                next_tag_type = "thinking"
                match = thinking_match
            elif answer_pos < thinking_pos and answer_pos < delegate_pos:
                next_tag_type = "answer"
                match = answer_match
            elif delegate_pos < thinking_pos and delegate_pos < answer_pos:
                next_tag_type = "delegate"
                match = delegate_match
            else:
                # No more tags found
                break

            # Process the found tag
            if next_tag_type == "thinking":
                thought = match.group(1).strip()
                thoughts.append(thought)
                # Trigger callback for thinking
                self._safe_event_handler(event_name="on_thinking", text=thought)
            elif next_tag_type == "answer":
                answer = match.group(1).strip()
                answers.append(answer)
                # Trigger callback for answer
                self._safe_event_handler(event_name="on_answer", text=answer)
            elif next_tag_type == "delegate":
                delegate_content = match.group(1).strip()
                agent_name, query = self._parse_delegate_tag(delegate_content)
                if agent_name:
                    # Execute the delegate and get result
                    result = self._execute_delegate(agent_name, query)
                    delegates.append((agent_name, query, result))

            # Move past the processed tag
            remaining_content = remaining_content[match.end() :]

        # If no matches are found, return empty lists
        if not thoughts and not answers and not delegates:
            return [], [], []

        return thoughts, answers, delegates

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

        # Initialize return values
        content = ""
        thinking = []
        answer = []
        tool_used = False
        tool_name = None
        tool_parameters = {}
        tool_results = []
        tool_data = None
        delegate_used = False
        delegate_data = []

        # Initialize the chronological event list for this iteration
        # This will be populated by _safe_event_handler
        self._current_iteration_events = []

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
            # Process the content to handle thinking/answer/delegate tags
            # The _process_content method will call event handlers which populate _current_iteration_events
            thinking, answer, delegates = self._process_content(content)

            # Check if delegation occurred
            if delegates:
                delegate_used = True
                delegate_data = delegates

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

                    # Call the observation callback which will add to _current_iteration_events
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

                    # Log the error which will add to _current_iteration_events
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

                # Log the error which will add to _current_iteration_events
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

            # Store the events list before returning
            iteration_events = self._current_iteration_events
            delattr(self, "_current_iteration_events")

            return {
                "content": content,
                "thinking": thinking,
                "answer": answer,
                "tool_used": True,
                "tool_data": tool_data,
                "tool_results": tool_results,
                "delegate_used": delegate_used,
                "delegate_data": delegate_data,
                "events": iteration_events,
            }

        # Store the events list before returning
        iteration_events = self._current_iteration_events
        delattr(self, "_current_iteration_events")

        return {
            "content": content,
            "thinking": thinking,
            "answer": answer,
            "tool_used": False,
            "tool_data": None,
            "tool_results": tool_results,
            "delegate_used": delegate_used,
            "delegate_data": delegate_data,
            "events": iteration_events,
        }

    def invoke(self, prompt: str, max_iterations: int = 10) -> dict[str, Any]:
        """
        Run the agent with the given prompt using the model's invoke method.

        :param prompt: The initial prompt to feed to the LLM
        :param max_iterations: Maximum number of model-tool-model iterations
        :return: A dictionary containing 'content', 'thinking', 'tool_use', 'delegate_use', 'answer', and 'events'
        """
        # Clear or reset the agent's memory context
        self._flush_memory()

        # Add the user's prompt to memory
        self.memory.add_message(role="user", content=prompt)

        # Initialize buffers for thinking and answer
        all_thinking = []
        all_answer = []
        all_tool_use = []
        all_delegate_use = []
        full_response = []
        # Initialize a list to store all events in chronological order
        all_events = []
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

            # Add structured delegate data if delegation occurred
            if response["delegate_used"] and response["delegate_data"]:
                all_delegate_use.extend(response["delegate_data"])

            # Collect events from this iteration in chronological order
            if "events" in response:
                all_events.extend(response["events"])

            # Add to the full response if we got a valid response
            if response:
                full_response.append(response["content"])

            # Add tool result messages to the full response
            if response["tool_results"]:
                full_response.extend(response["tool_results"])

            # Add delegate results to the full response
            if response["delegate_used"] and response["delegate_data"]:
                for agent_name, query, result in response["delegate_data"]:
                    full_response.append(
                        f"[Delegate Result] {agent_name}({query}) -> {result}"
                    )

            # If no tool was used and no delegation occurred, we're done
            # IMPORTANT: This check needs to ensure we don't exit early during delegation chains
            if not response["tool_used"] and not response["delegate_used"]:
                # Check if we have an answer before breaking
                if not all_answer:
                    # If we've delegated but don't yet have an answer, continue for at least one more iteration
                    if all_delegate_use and iterations < max_iterations:
                        continue
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
            "delegate_use": all_delegate_use,
            "answer": answer,
            "events": all_events,
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
        - delegate_use: List of tuples with agent name, query, and result
        - answer: Final answer text (combined)

        :param events: The events to unpack
        :return: Dictionary with standardized output format
        """
        # Initialize result containers
        thinking = []
        answer_chunks = []
        tool_use = []
        delegate_use = []
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

            elif event_type == "delegate_usage":
                # Extract delegate data and add to delegate_use list
                delegate_data = event["content"]
                delegate_use.append(
                    (
                        delegate_data["agent_name"],
                        delegate_data["query"],
                        delegate_data["result"],
                    )
                )

                # Add formatted delegate usage to full content
                delegate_message = f"[Delegate Result] {delegate_data['agent_name']}({delegate_data['query']}) -> {delegate_data['result']}"
                full_content.append(delegate_message)

        # Combine answer chunks into a single answer
        answer = "".join(answer_chunks) if answer_chunks else None

        # Return a dictionary with the same structure as invoke() method
        return {
            "content": "\n\n".join(full_content),
            "thinking": thinking,
            "tool_use": tool_use,
            "delegate_use": delegate_use,
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
        delegate_used = False

        # State tracking for tool use - initialize only what's needed
        current_tool_data = {}
        is_collecting_tool_data = False
        current_tool_name = None
        current_tool_buffer = ""

        # Create a stream handler for processing tagged content
        def handle_tagged_content(tag_name: str, content: str) -> None:
            """Handle content from tagged sections.

            :param tag_name: Name of the tag (thinking, answer, delegate, etc)
            :param content: Content within the tag
            """
            nonlocal iteration_response, delegate_used

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
            elif tag_name == "delegate":
                delegate_used = True
                agent_name, query = self._parse_delegate_tag(content)
                if agent_name:
                    result = self._execute_delegate(agent_name, query)
                    # Store the delegate event
                    stream_handler.events.append(
                        {
                            "type": "delegate_usage",
                            "content": {
                                "agent_name": agent_name,
                                "query": query,
                                "result": result,
                            },
                        }
                    )

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
                if iteration_response and not tool_used and not delegate_used:
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

        # Return tool_used or delegate_used as a hidden value for the stream method
        if tool_used:
            # This is a special marker that will not be displayed but will be detected by the stream method
            yield "[[TOOL_USED]]"
        elif delegate_used:
            # This is a special marker for delegation
            yield "[[DELEGATE_USED]]"

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
        # Track delegation and answer tracking
        has_answer = False
        has_delegated = False

        while iterations < max_iterations:
            iterations += 1

            # Get current messages and system
            messages = self.memory.get_messages()
            system = self.memory.get_system_message()

            # Prep messages for the model
            prompt_obj = Messages(system=system, messages=messages)

            # Stream and process the response, tracking if a tool or delegate was used
            tool_used = False
            delegate_used = False

            for chunk in self._stream_iteration(prompt_obj):
                # Check if this is our special marker for tool usage
                if chunk == "[[TOOL_USED]]":
                    tool_used = True
                    # Don't yield this special marker
                    continue
                # Check if this is our special marker for delegate usage
                elif chunk == "[[DELEGATE_USED]]":
                    delegate_used = True
                    has_delegated = True
                    # Don't yield this special marker
                    continue

                # Check if this is our data collection chunk
                if isinstance(chunk, dict):
                    # Collect the events
                    all_events.extend(chunk["events"])

                    # Check if any of the events are answer events
                    for event in chunk.get("events", []):
                        if event["type"] == "answer" and event["content"]:
                            has_answer = True

                    # Don't yield the data collection chunk
                    continue

                # Yield all other chunks normally
                yield chunk

            # If no tool was used and no delegation occurred, we're done
            # IMPORTANT: This check needs to ensure we don't exit early during delegation chains
            if not tool_used and not delegate_used:
                # If we've delegated but don't yet have an answer, continue for at least one more iteration
                if has_delegated and not has_answer and iterations < max_iterations:
                    continue
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
        :return: If stream=False: A dictionary containing 'content', 'thinking', 'tool_use', 'delegate_use', 'answer', and 'events'
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

    def update_user_system_message(self, new_system_message: str | None) -> None:
        """Update the user-provided portion of the system message.

        :param new_system_message: New system message text, or None to clear it
        """
        self._user_system_message = new_system_message

        # Rebuild system message with the new user component
        self._system_message = self._build_system_message()
        self.memory.set_system_message(self._system_message)

        if new_system_message:
            logger.info("Updated user system message")
        else:
            logger.info("Cleared user system message")


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

    # # Example 1: Using BedrockAgent with multiple tools
    # logger.info("\nExample 1: Using BedrockAgent with multiple tools:")
    # setup_logging(log_level="DEBUG", are_you_serious=False)

    # # Define a event handler class to avoid using globals
    # class DummyEventHandler:
    #     """Event handler for managing agent event handlers with internal state."""

    #     def on_tool_use(self, tool_name, parameters, result):
    #         """Handle tool use events."""
    #         print(f"🔧 TOOL: {tool_name} with {parameters} -> {result}")

    #     def on_thinking(self, text):
    #         """Handle agent thinking events."""
    #         print(f"🧠 THINKING: {text}")

    #     def on_answer(self, text):
    #         print(f"💬 ANSWER: {text}")

    # # Create the event handler instance
    # event_handler = DummyEventHandler()

    # agent = BedrockAgent(
    #     identifier="WeatherAssistant",
    #     model=Claude35Sonnet(),
    #     description="An assistant that can provide weather information and perform temperature conversions",
    #     tools=[get_weather, convert_temperature],
    #     memory=FleetingMemory(),
    #     log_agentic_response=True,
    #     system_message="Be very courteous.",
    #     event_handlers={
    #         "on_tool_use": event_handler.on_tool_use,
    #         "on_thinking": event_handler.on_thinking,
    #         "on_answer": event_handler.on_answer,
    #     },
    # )

    # response = agent.run(prompt, stream=False)

    # from pprint import pprint
    # pprint(response)

    #############
    # Streaming #
    #############

    # # Example 2: Using BedrockAgent with streaming
    # logger.info("\nExample 2: Using BedrockAgent with streaming:")

    # # Create a new agent with streaming enabled
    # agent = BedrockAgent(
    #     identifier="WeatherAssistant",
    #     model=NovaPro(region="us-east-1"),
    #     description="An assistant that can provide weather information and perform temperature conversions",
    #     tools=[get_weather, convert_temperature],
    #     memory=FleetingMemory(),
    #     log_agentic_response=True,
    #     system_message="Answer like a pirate",
    # )

    # # Use streaming mode
    # collected_data = agent.run(prompt, stream=True)

    # # Access the answer
    # print(f"Answer: {collected_data['answer']}")

    #########################
    # Prompt model directly #
    #########################

    # # Example 3: Custom event handler
    # import random
    # from fence.agents.bedrock import EventHandler

    # def on_tool_use(tool_name, parameters, result):
    #     """Handle tool use events."""
    #     print(
    #         f"SENDING TOOL USE TO SLACK: CALLED A TOOL: {tool_name} with {parameters} -> {result}"
    #     )

    # def on_thinking(text):
    #     """Handle agent thinking events."""
    #     synonyms_for_thinking = [
    #         "thinking",
    #         "pondering",
    #         "considering",
    #         "evaluating",
    #         "analyzing",
    #         "reflecting",
    #     ]
    #     print(f"SENDING THINKING TO SLACK: *{random.choice(synonyms_for_thinking)}*")

    # def on_answer(text):
    #     """Handle agent answer events."""
    #     print(f"SENDING ANSWER TO SLACK: {text}")

    # event_handler = EventHandler(
    #     on_tool_use=on_tool_use, on_thinking=on_thinking, on_answer=on_answer
    # )

    # # Create the agent
    # agent = BedrockAgent(model=Claude35Sonnet(), event_handlers=event_handler)

    # # Run the agent
    # agent.run("What is the weather in New York, in Celsius?")

    #########################
    # Agent Delegation Demo #
    #########################

    logger.info("\nExample 4: Using BedrockAgent with dynamic delegation:")
    setup_logging(log_level="INFO", are_you_serious=False)

    # Define event handler for delegation events

    # Create a parent agent initially without delegates
    parent_agent = BedrockAgent(
        identifier="ParentAgent",
        model=Claude35Sonnet(),
        description="A parent agent that can delegate tasks to specialist agents",
        memory=FleetingMemory(),
        log_agentic_response=True,
        system_message="You are a helpful assistant that loves to collaborate with specialists.",
    )

    # Create specialist weather agent with tools
    weather_specialist = BedrockAgent(
        identifier="WeatherSpecialist",
        model=Claude35Sonnet(),
        description="A specialist agent for weather information and temperature conversions",
        tools=[get_weather, convert_temperature],
        memory=FleetingMemory(),
        log_agentic_response=True,
        system_message="You are a weather expert. Provide precise and accurate weather information.",
    )

    # Create math specialist agent
    math_specialist = BedrockAgent(
        identifier="MathSpecialist",
        model=Claude35Sonnet(),
        description="A specialist agent for complex mathematical calculations",
        memory=FleetingMemory(),
        log_agentic_response=True,
        system_message="You are a math expert. Solve math problems with careful step-by-step work.",
    )

    # Add delegates
    parent_agent.add_delegate(weather_specialist)
    parent_agent.add_delegate(math_specialist)

    # Run with a different query that could use either delegate
    print("\nRunning agent with math and weather query:")
    result = parent_agent.run(
        "If it's 75°F in New York and 68°F in London, what's the square root of the average temperature in Celsius?"
    )
    print(f"Answer: {result['answer']}")
