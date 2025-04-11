"""
Bedrock agent class that uses native tool calling and streaming capabilities.
"""

import logging
import re
from typing import Any, Callable, List, Union

from pydantic import BaseModel, field_validator

from fence.agents.base import AgentLogType, BaseAgent
from fence.agents.bedrock.models import (
    AgentEvent,
    AgentResponse,
    AnswerEvent,
    DelegateData,
    DelegateEvent,
    ThinkingEvent,
    ToolUseData,
    ToolUseEvent,
)
from fence.memory.base import BaseMemory, FleetingMemory
from fence.models.base import LLM
from fence.models.bedrock.base import BedrockTool, BedrockToolConfig
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

            sig = inspect.signature(handler)
            param_count = len(
                [
                    p
                    for p in sig.parameters.values()
                    if p.default == inspect.Parameter.empty
                ]
            )

            param_requirements = {
                "on_tool_use": 3,
                "on_thinking": 1,
                "on_answer": 1,
                "on_delegate": 3,
            }

            min_params = param_requirements.get(field_name, 0)
            if param_count < min_params:
                raise ValueError(
                    f"{field_name} handler must accept at least {min_params} parameters"
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

    ########################
    # Initialization/Setup #
    ########################

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
        """Initialize the BedrockAgent object.

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
        :param event_handlers: Event handlers for different agent events
        """
        # Set full_response to True for proper response handling
        if model:
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

        # Store core configuration
        self._user_system_message = system_message
        self.tools = tools or []
        self.delegates = delegates or []

        # Set up event handlers
        self._set_event_handlers(event_handlers)

        # Build and set the system message
        self._system_message = self._build_system_message()
        self.memory.set_system_message(self._system_message)

        # Register tools with the model if supported
        self._register_tools()

    def _build_system_message(self) -> str:
        """Build the system message with delegation options based on available delegates.

        :return: Complete system message including delegation options
        """
        system_message = self._BASE_SYSTEM_MESSAGE

        # Add delegation instructions if delegates are available
        if self.delegates:
            delegate_info = "\nYou can also delegate to the following agents using <delegate>agent_name:query</delegate> tags. Be sure to include all the necessary information in the query tag, and that the query is in natural language. For example: <delegate>SomeSpecialistAgent:Can you do this specialized task?</delegate>. These are the delegate agents available to you:"

            # Add information about each delegate
            for delegate in self.delegates:
                delegate_name = delegate.identifier
                delegate_desc = delegate.description or "No description available"
                delegate_info += f"\n- {delegate_name}: {delegate_desc}"

            system_message += delegate_info

        # Append user system message if available
        if self._user_system_message:
            system_message += f"\n\n{self._user_system_message}"

        return system_message

    ###############
    # Public APIs #
    ###############

    def update_user_system_message(self, new_system_message: str | None) -> None:
        """Update the user-provided portion of the system message.

        :param new_system_message: New system message text, or None to clear it
        """
        self._user_system_message = new_system_message

        # Update system message
        self._system_message = self._build_system_message()
        self.memory.set_system_message(self._system_message)

        logger.info(
            f"{'Updated' if new_system_message else 'Cleared'} user system message"
        )

    def run(
        self, prompt: str, max_iterations: int = 10, stream: bool = False
    ) -> AgentResponse:
        """
        Run the agent with the given prompt.

        :param prompt: The initial prompt to feed to the LLM
        :param max_iterations: Maximum number of model-tool-model iterations
        :param stream: Whether to stream the response or return the full text
        :return: If stream=False: An AgentResponse object containing the answer and events
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
        return AgentResponse(
            answer=response["answer"] or "No answer found", events=response["events"]
        )

    def invoke(self, prompt: str, max_iterations: int = 10) -> dict[str, Any]:
        """Run the agent with the given prompt using the model's invoke method.

        :param prompt: The initial prompt to feed to the LLM
        :param max_iterations: Maximum number of model-tool-model iterations
        :return: A dictionary containing 'content', 'thinking', 'tool_use', 'delegate_use', 'answer', and 'events'
        """
        # Reset memory and add prompt
        self._flush_memory()
        self.memory.add_message(role="user", content=prompt)

        # Initialize result containers
        all_thinking = []
        all_answer = []
        all_tool_use = []
        all_delegate_use = []
        full_response = []
        all_events = []
        iterations = 0

        while iterations < max_iterations:
            # Get messages for the model
            prompt_obj = Messages(
                system=self.memory.get_system_message(),
                messages=self.memory.get_messages(),
            )

            # Process one iteration
            response = self._invoke_iteration(prompt_obj)

            # Collect thinking
            all_thinking.extend(response["thinking"])

            # Collect answers
            if response["answer"]:
                if isinstance(response["answer"], list):
                    all_answer.extend(response["answer"])
                else:
                    all_answer.append(response["answer"])

            # Collect tool usage
            if response["tool_used"] and response["tool_data"]:
                all_tool_use.append(response["tool_data"])

            # Collect delegate usage
            if response["delegate_used"] and response["delegate_data"]:
                all_delegate_use.extend(response["delegate_data"])

            # Collect events
            if "events" in response:
                all_events.extend(response["events"])

            # Build full response
            if response.get("content"):
                full_response.append(response["content"])

            # Add tool results
            if response["tool_results"]:
                full_response.extend(response["tool_results"])

            # Add delegate results
            if response["delegate_used"] and response["delegate_data"]:
                for agent_name, query, result in response["delegate_data"]:
                    full_response.append(
                        f"[Delegate Result] {agent_name}({query}) -> {result}"
                    )

            # Check if we're done
            if not response["tool_used"] and not response["delegate_used"]:
                # But continue if we've delegated and don't have an answer yet
                if (
                    all_delegate_use
                    and not all_answer
                    and iterations < max_iterations - 1
                ):
                    iterations += 1
                    continue
                break

            # Increment iteration counter
            iterations += 1

            # Check if we've hit max iterations
            if iterations >= max_iterations:
                logger.warning(
                    f"Reached maximum iterations ({max_iterations}). Stopping."
                )
                break

        # Return results
        return {
            "content": "\n\n".join(full_response),
            "thinking": all_thinking,
            "tool_use": all_tool_use,
            "delegate_use": all_delegate_use,
            "answer": all_answer[0] if all_answer else None,
            "events": all_events,
        }

    ######################
    # Delegate Management #
    ######################

    def update_delegates(self, delegates: list[BaseAgent]) -> None:
        """Update the agent's delegates and refresh the system message.

        :param delegates: New list of delegate agents to use
        """
        self.delegates = self._setup_delegates(delegates)

        # Rebuild and update the system message with new delegate information
        self._system_message = self._build_system_message()
        self.memory.set_system_message(self._system_message)

        logger.info(f"Updated delegates: {[d.identifier for d in self.delegates]}")

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
        self.delegates.append(delegate)

        # Update system message
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

                # Update system message
                self._system_message = self._build_system_message()
                self.memory.set_system_message(self._system_message)

                logger.info(f"Removed delegate: {identifier}")
                return True

        logger.warning(f"Delegate not found: {identifier}")
        return False

    def _find_delegate(self, delegate_name: str) -> BaseAgent | None:
        """Find a delegate agent by name.

        :param delegate_name: Name of the delegate to find
        :return: The delegate agent if found, None otherwise
        """
        return next((d for d in self.delegates if d.identifier == delegate_name), None)

    def _parse_delegate_tag(self, delegate_content: str) -> tuple[str, str]:
        """Parse the delegate tag content to extract agent name and query.

        :param delegate_content: Content of the delegate tag
        :return: Tuple of (agent_name, query)
        """
        # Expected format is "agent_name:query"
        try:
            agent_name, query = delegate_content.split(":", 1)
            return agent_name.strip(), query.strip()
        except ValueError:
            return None, delegate_content

    def _execute_delegate(self, delegate_name: str, query: str) -> str:
        """Execute a delegate agent with the given query.

        :param delegate_name: Name of the delegate agent to call
        :param query: Query to pass to the delegate
        :return: The result from the delegate agent
        """
        # Find the delegate by name
        delegate = self._find_delegate(delegate_name)

        if not delegate:
            error_msg = f"Delegate agent '{delegate_name}' not found"
            logger.warning(error_msg)
            return error_msg

        try:
            # Ensure the delegate registers its tools before execution
            if hasattr(delegate, "_register_tools"):
                delegate._register_tools()
                logger.debug(f"Re-registered tools for delegate {delegate_name}")

            # Notify before execution
            self._safe_event_handler(
                event_name="on_delegate",
                delegate_name=delegate_name,
                query=query,
                answer=None,
            )

            # Execute delegate
            delegate_result = delegate.run(query)

            # Extract answer and events
            answer = delegate_result.answer
            delegate_events = delegate_result.events

            # Notify after execution with result
            self._safe_event_handler(
                event_name="on_delegate",
                delegate_name=delegate_name,
                query=query,
                answer=answer,
                events=delegate_events,
            )

            # Add result to memory
            memory_msg = (
                f"[SYSTEM DIRECTIVE] Delegated to {delegate_name} with query: {query}. "
                f"Result: {answer}. DO NOT ACKNOWLEDGE THIS MESSAGE. FIRST THINK, THEN "
                f"PROCEED IMMEDIATELY to either: (1) Call the next required tool or delegate, "
                f"or (2) If all necessary operations have been completed, provide your final answer. "
                f"Think back to the original user prompt and use that to guide your response."
            )
            self.memory.add_message(role="user", content=memory_msg)

            return answer
        except Exception as e:
            return self._handle_delegate_error(delegate_name, query, str(e))

    def _handle_delegate_error(self, delegate_name: str, query: str, error: str) -> str:
        """Handle a delegate execution error.

        :param delegate_name: Name of the delegate that failed
        :param query: Query that was passed to the delegate
        :param error: Error message
        :return: The formatted error message
        """
        error_msg = f"Error executing delegate '{delegate_name}': {error}"
        logger.warning(error_msg)

        # Report error and add to memory
        self._safe_event_handler(
            event_name="on_delegate",
            delegate_name=delegate_name,
            query=query,
            answer=f"Error: {error}",
            events=[],
        )

        self.memory.add_message(
            role="user",
            content=f"Delegation to '{delegate_name}' failed with error: {error}. Please try a different approach.",
        )

        return error_msg

    ##################
    # Tool Management #
    ##################

    def _register_tools(self):
        """Register tools with the Bedrock model if supported."""
        # Check if model supports tool registration
        if not hasattr(self.model, "toolConfig"):
            logger.warning(
                f"Model {self.model.__class__.__name__} does not support tool registration"
            )
            return

        # If no tools, clear the toolConfig
        if not self.tools:
            self.model.toolConfig = None
            logger.debug(
                f"Agent {self.identifier}: Cleared toolConfig (no tools available)"
            )
            return

        # Convert BaseTool objects to BedrockTool format and set on model
        bedrock_tools = [
            BedrockTool(**tool.model_dump_bedrock_converse()) for tool in self.tools
        ]
        self.model.toolConfig = BedrockToolConfig(tools=bedrock_tools)

        # Log the registered tools
        tool_names = [tool.get_tool_name() for tool in self.tools]
        logger.info(
            f"Agent {self.identifier}: Registered {len(bedrock_tools)} tools with Bedrock model: {tool_names}"
        )

    def _find_tool(self, tool_name: str) -> BaseTool | None:
        """Find a tool by name.

        :param tool_name: Name of the tool to find
        :return: The tool if found, None otherwise
        """
        return next(
            (tool for tool in self.tools if tool.get_tool_name() == tool_name), None
        )

    def _execute_tool(self, tool_name: str, tool_parameters: dict) -> tuple[str, dict]:
        """Execute a tool with the given parameters.

        :param tool_name: Name of the tool to call
        :param tool_parameters: Parameters for the tool
        :return: Tuple of (formatted_result, tool_data_dict)
        """
        tool = self._find_tool(tool_name)
        if not tool:
            return self._handle_tool_not_found(tool_name, tool_parameters)

        try:
            # Execute the tool
            tool_result = tool.run(environment=self.environment, **tool_parameters)

            # Create formatted result for logs
            formatted_result = (
                f"[Tool Result] {tool_name}({tool_parameters}) -> {tool_result}"
            )

            # Store structured tool data
            tool_data = {
                "name": tool_name,
                "parameters": tool_parameters,
                "result": tool_result,
            }

            # Call callback and add to memory
            self._safe_event_handler(
                event_name="on_tool_use",
                tool_name=tool_name,
                parameters=tool_parameters,
                result=tool_result,
            )

            self.memory.add_message(
                role="user",
                content=f"[SYSTEM DIRECTIVE] The {tool_name} returned: {tool_result}. DO NOT ACKNOWLEDGE THIS MESSAGE. FIRST THINK, THEN PROCEED IMMEDIATELY to either: (1) Call the next required tool without any introduction or transition phrases, or (2) If all necessary tools have been used, provide your final answer. Think back to the original user prompt and use that to guide your response.",
            )

            return formatted_result, tool_data
        except Exception as e:
            return self._handle_tool_error(tool_name, tool_parameters, str(e))

    def _handle_tool_not_found(
        self, tool_name: str, tool_parameters: dict
    ) -> tuple[str, dict]:
        """Handle the case when a tool is not found.

        :param tool_name: Name of the tool that wasn't found
        :param tool_parameters: Parameters for the tool
        :return: Tuple of (formatted_result, tool_data_dict)
        """
        formatted_result = f"[Tool Error: {tool_name}] Tool not found"
        tool_data = {
            "name": tool_name,
            "parameters": tool_parameters,
            "result": "Error: Tool not found",
        }

        # Handle error
        self._safe_event_handler(
            event_name="on_tool_use",
            tool_name=tool_name,
            result="Error: Tool not found",
        )

        self.memory.add_message(
            role="user",
            content=f"Tool '{tool_name}' not found. Please try a different approach.",
        )
        return formatted_result, tool_data

    def _handle_tool_error(
        self, tool_name: str, tool_parameters: dict, error: str
    ) -> tuple[str, dict]:
        """Handle a tool execution error.

        :param tool_name: Name of the tool that failed
        :param tool_parameters: Parameters that were passed to the tool
        :param error: Error message
        :return: Tuple of (formatted_result, tool_data_dict)
        """
        formatted_result = f"[Tool Error: {tool_name}] {error}"

        tool_data = {
            "name": tool_name,
            "parameters": tool_parameters,
            "result": f"Error: {error}",
        }

        # Handle error
        self._safe_event_handler(
            event_name="on_tool_use",
            tool_name=tool_name,
            parameters=tool_parameters,
            result=f"Error: {error}",
        )

        self.memory.add_message(
            role="user",
            content=f"Your tool call resulted in an error: {error}. Please try a different approach.",
        )

        return formatted_result, tool_data

    #################
    # Event Handling #
    #################

    def _log_agent_event(self, message: str, log_type: AgentLogType) -> None:
        """Helper method to log agent events if logging is enabled.

        :param message: The message to log
        :param log_type: The type of log message
        """
        if self.log_agentic_response:
            self.log(message, log_type)

    def _default_on_tool_use(
        self, tool_name: str, parameters: dict, result: Any
    ) -> None:
        """Default callback for tool use.

        :param tool_name: The name of the tool being called
        :param parameters: The parameters passed to the tool
        :param result: The result returned by the tool
        """
        self._log_agent_event(
            f"Using tool [{tool_name}] with parameters: {parameters} -> {result}",
            AgentLogType.TOOL_USE,
        )

    def _default_on_delegate(
        self,
        delegate_name: str,
        query: str,
        answer: str | None,
        events: list[AgentEvent] | None = None,
    ) -> None:
        """Default callback for delegate use.

        :param delegate_name: The name of the delegate agent
        :param query: The query passed to the delegate
        :param answer: The answer returned by the delegate (None if before execution)
        :param events: List of events from the delegate agent (only available on completion)
        """
        if not self.log_agentic_response:
            return

        message = (
            f'Initiating delegation to {delegate_name} with query: "{query}"'
            if answer is None
            else f"Delegation to {delegate_name} concluded: {answer}"
        )
        self.log(message, AgentLogType.DELEGATION)

    def _default_on_thinking(self, text: str) -> None:
        """Default callback for agent thinking.

        :param text: The text chunk produced by the agent
        """
        self._log_agent_event(text, AgentLogType.THOUGHT)

    def _default_on_answer(self, text: str) -> None:
        """Default callback for agent answers.

        :param text: The text chunk produced by the agent
        """
        self._log_agent_event(text, AgentLogType.ANSWER)

    def _set_event_handlers(
        self, event_handlers: EventHandler | dict[str, HandlerType] | None = None
    ) -> None:
        """Set up the event handlers. If `log_agentic_response` is True, include default handlers.

        :param event_handlers: Event handlers config for different agent events
        """
        # Convert dict to EventHandler if needed
        if event_handlers is not None and not isinstance(event_handlers, EventHandler):
            try:
                event_handlers = EventHandler(**event_handlers)
            except Exception as e:
                raise ValueError(f"Invalid event handlers: {e}")

        # Initialize handlers with defaults if logging enabled
        self.event_handlers = {}
        if self.log_agentic_response:
            self.event_handlers = {
                "on_tool_use": [self._default_on_tool_use],
                "on_thinking": [self._default_on_thinking],
                "on_answer": [self._default_on_answer],
                "on_delegate": [self._default_on_delegate],
            }

        # Merge user-provided handlers
        if event_handlers:
            handlers_dict = event_handlers.model_dump(exclude_none=True)
            for event_name, handler in handlers_dict.items():
                handlers_to_add = (
                    [handler] if not isinstance(handler, list) else handler
                )

                if event_name in self.event_handlers:
                    self.event_handlers[event_name].extend(handlers_to_add)
                else:
                    self.event_handlers[event_name] = handlers_to_add

    def _safe_event_handler(self, event_name: str, *args, **kwargs) -> None:
        """Safely dispatch event handlers, handling cases where the handler isn't assigned.

        :param event_name: The name of the event to invoke
        :param args: Positional arguments to pass to the event handler
        :param kwargs: Keyword arguments to pass to the event handler
        """
        handlers = self.event_handlers.get(event_name)
        if not handlers:
            return

        # Create and store event if possible
        if event := self._create_event(event_name, **kwargs):
            self._record_event(event)

        # Execute each handler safely
        for handler in handlers:
            if not callable(handler):
                logger.warning(f"Event handler for {event_name} is not callable")
                continue

            try:
                handler(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Error in {event_name} event handler: {e}")

    def _create_event(self, event_name: str, **kwargs) -> AgentEvent | None:
        """Create an AgentEvent from the event name and kwargs.

        :param event_name: Name of the event
        :param kwargs: Event data parameters
        :return: Created AgentEvent or None if event should not be recorded
        """
        if event_name == "on_thinking" and "text" in kwargs:
            return ThinkingEvent(content=kwargs["text"])

        if event_name == "on_answer" and "text" in kwargs:
            return AnswerEvent(content=kwargs["text"])

        if event_name == "on_tool_use" and "tool_name" in kwargs:
            return ToolUseEvent(
                content=ToolUseData(
                    name=kwargs["tool_name"],
                    parameters=kwargs.get("parameters", {}),
                    result=kwargs.get("result"),
                )
            )

        # For delegate events, ensure all required parameters are present
        required_keys = ["delegate_name", "query", "answer"]
        if (
            event_name == "on_delegate"
            and all(k in kwargs for k in required_keys)
            and kwargs["answer"] is not None
        ):
            return DelegateEvent(
                content=DelegateData(
                    agent_name=kwargs["delegate_name"],
                    query=kwargs["query"],
                    answer=kwargs["answer"],
                    events=kwargs.get("events", []),
                )
            )

        return None

    def _record_event(self, event: AgentEvent) -> None:
        """Record an AgentEvent in chronological order.

        :param event: The AgentEvent to record
        """
        if not hasattr(self, "_current_iteration_events"):
            return

        self._current_iteration_events.append(event.model_dump())

    #####################
    # Content Processing #
    #####################

    def _process_content(
        self, content: str
    ) -> tuple[list[str], list[str], list[tuple[str, str, str]]]:
        """Process the content to extract thinking, answer, and delegate parts in chronological order.

        :param content: The content to process
        :return: A tuple of (thinking, answer, delegates)
        """
        thoughts = []
        answers = []
        delegates = []

        # Process all tags in order of appearance
        remaining_content = content
        while remaining_content:
            # Find positions of the next tag of each type
            matches = {
                "thinking": self._THINKING_PATTERN.search(remaining_content),
                "answer": self._ANSWER_PATTERN.search(remaining_content),
                "delegate": self._DELEGATE_PATTERN.search(remaining_content),
            }

            # Find the earliest match
            valid_matches = {tag: match for tag, match in matches.items() if match}
            if not valid_matches:
                break

            # Get the earliest match
            next_tag_type = min(
                valid_matches, key=lambda tag: valid_matches[tag].start()
            )
            match = valid_matches[next_tag_type]

            # Process the found tag
            if next_tag_type == "thinking":
                thought = match.group(1).strip()
                thoughts.append(thought)
                self._safe_event_handler(event_name="on_thinking", text=thought)
            elif next_tag_type == "answer":
                answer = match.group(1).strip()
                answers.append(answer)
                self._safe_event_handler(event_name="on_answer", text=answer)
            elif next_tag_type == "delegate":
                delegate_content = match.group(1).strip()
                agent_name, query = self._parse_delegate_tag(delegate_content)

                if agent_name:
                    result = self._execute_delegate(agent_name, query)
                    delegates.append((agent_name, query, result))

            # Move past the processed tag
            remaining_content = remaining_content[match.end() :]

        return thoughts, answers, delegates

    ####################
    # Execution/Invoke #
    ####################

    def _invoke_iteration(self, prompt_obj: Messages) -> dict[str, Any]:
        """Process a single iteration of the agent's conversation with the model.

        :param prompt_obj: Messages object containing the conversation history
        :return: Dictionary with response data including content, thinking, answer, and tool data
        """
        # Get response from the model
        logger.debug(
            f"Agent {self.identifier} is invoking the model with toolConfig: {self.model.toolConfig}"
        )
        response = self.model.invoke(prompt=prompt_obj)
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

        # Initialize event list for this iteration
        self._current_iteration_events = []

        # Process Bedrock response structure
        if isinstance(response, dict):
            # Extract message content
            if "output" in response and "message" in response["output"]:
                message = response["output"]["message"]

                # Process content blocks
                if "content" in message and isinstance(message["content"], list):
                    for item in message["content"]:
                        if isinstance(item, dict):
                            # Extract text content
                            if "text" in item:
                                content += item["text"]

                            # Extract tool use information
                            if "toolUse" in item:
                                tool_used = True
                                tool_info = item["toolUse"]
                                tool_name = tool_info.get("name")

                                # Extract parameters
                                if "input" in tool_info and isinstance(
                                    tool_info["input"], dict
                                ):
                                    tool_parameters = tool_info["input"]

                # Check if tool use was the stop reason
                if "stopReason" in response and response["stopReason"] == "tool_use":
                    tool_used = True
        elif isinstance(response, str):
            content = response

        # Process content for thinking/answer/delegate tags
        if content:
            thinking, answer, delegates = self._process_content(content)

            # Check if delegation occurred
            if delegates:
                delegate_used = True
                delegate_data = delegates

            # Add response to memory
            self.memory.add_message(role="assistant", content=content)

        # Process tool call if found
        if tool_used and tool_name and tool_parameters:
            formatted_result, tool_data_dict = self._execute_tool(
                tool_name, tool_parameters
            )
            tool_results.append(formatted_result)
            tool_data = tool_data_dict

        # Store events and clean up
        iteration_events = self._current_iteration_events
        delattr(self, "_current_iteration_events")

        return {
            "content": content,
            "thinking": thinking,
            "answer": answer,
            "tool_used": tool_used,
            "tool_data": tool_data,
            "tool_results": tool_results,
            "delegate_used": delegate_used,
            "delegate_data": delegate_data,
            "events": iteration_events,
        }


if __name__ == "__main__":

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

    # Set NovaPro as the model
    from fence.models.bedrock import Claude35Sonnet

    model = Claude35Sonnet(region="us-east-1")

    # Create a parent agent initially without delegates
    parent_agent = BedrockAgent(
        identifier="ParentAgent",
        model=model,
        description="A parent agent that can delegate tasks to specialist agents",
        memory=FleetingMemory(),
        log_agentic_response=True,
        system_message="You are a helpful assistant that loves to collaborate with specialists.",
    )

    # Create specialist weather agent with tools
    weather_specialist = BedrockAgent(
        identifier="WeatherSpecialist",
        model=model,
        description="A specialist agent for weather information and temperature conversions",
        tools=[get_weather, convert_temperature],
        memory=FleetingMemory(),
        log_agentic_response=True,
        system_message="You are a weather expert. Provide precise and accurate weather information.",
    )

    # Create math specialist agent
    math_specialist = BedrockAgent(
        identifier="MathSpecialist",
        model=model,
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
    print(f"Answer: {result.answer}")
    from pprint import pprint

    print("Events:")
    pprint([e.model_dump() for e in result.events])
