"""
Bedrock agent class that uses native tool calling and streaming capabilities.
"""

import logging
import re
from pprint import pformat
from typing import Any, Callable, List, Union

from pydantic import BaseModel, ConfigDict, field_validator

from fence.agents.base import AgentLogType, BaseAgent
from fence.agents.bedrock.models import (
    AgentEvent,
    AgentEventTypes,
    AgentResponse,
    AgentStartEvent,
    AgentStopEvent,
    AnswerEvent,
    DelegateData,
    DelegateStartEvent,
    DelegateStopEvent,
    ThinkingEvent,
    ToolUseData,
    ToolUseStartEvent,
    ToolUseStopEvent,
)
from fence.memory.base import BaseMemory
from fence.models.base import LLM
from fence.models.bedrock.base import BedrockTool, BedrockToolConfig
from fence.models.bedrock.claude import Claude35Sonnet
from fence.models.bedrock.nova import NovaPro
from fence.templates.models import (
    Content,
    Messages,
    TextContent,
    ToolResultBlock,
    ToolResultContent,
    ToolResultContentBlockJson,
    ToolResultContentBlockText,
    ToolUseBlock,
    ToolUseContent,
)
from fence.tools.base import BaseTool

logger = logging.getLogger(__name__)


HandlerType = Union[Callable, List[Callable]]


class EventHandlers(BaseModel):
    """Event handlers for the BedrockAgent.

    :param on_start: Called when the agent starts
    :param on_tool_use_start: Called when the agent starts using a tool
    :param on_tool_use_stop: Called when the agent completes using a tool
    :param on_thinking: Called when the agent is thinking
    :param on_answer: Called when the agent provides text answer chunks
    :param on_delegation_start: Called when the agent starts delegating to another agent
    :param on_delegation_stop: Called when the agent stops delegating to another agent
    :param on_stop: Called when the agent stops
    """

    on_start: HandlerType | None = None
    on_tool_use_start: HandlerType | None = None
    on_tool_use_stop: HandlerType | None = None
    on_thinking: HandlerType | None = None
    on_answer: HandlerType | None = None
    on_delegation_start: HandlerType | None = None
    on_delegation_stop: HandlerType | None = None
    on_stop: HandlerType | None = None

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
                "on_start": [0, {}],
                "on_tool_use_start": [2, {"tool_name", "parameters"}],
                "on_tool_use_stop": [3, {"tool_name", "parameters", "result"}],
                "on_thinking": [1, {"text"}],
                "on_answer": [1, {"text"}],
                "on_delegation_start": [2, {"delegate_name", "query"}],
                "on_delegation_stop": [
                    4,
                    {"delegate_name", "query", "answer", "events"},
                ],
                "on_stop": [0, {}],
            }

            # Check if the handler has the correct number of parameters
            min_params = param_requirements.get(field_name)[0]
            if param_count < min_params:
                raise ValueError(
                    f"{field_name} handler must accept at least {min_params} parameters"
                )

            # Check if the handler has the correct parameter names (order not important)
            if param_count > 0:
                required_params = param_requirements.get(field_name)[1]
                given_params = {p.name for p in sig.parameters.values()}

                # Calculate superfluous and missing parameters
                superfluous_params = given_params - required_params
                missing_params = required_params - given_params

                # Check if the required parameters are present in the given parameters
                if given_params != set(required_params):
                    raise ValueError(
                        f"{field_name} handler parameter mismatch: missing {missing_params if missing_params else 'none'}, superfluous {superfluous_params if superfluous_params else 'none'}"
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

    model_config = ConfigDict(extra="forbid")


class BedrockAgent(BaseAgent):
    """
    Bedrock agent that uses native tool calling and streaming capabilities.
    """

    _BASE_SYSTEM_MESSAGE = """
You are a helpful assistant. You can think in <thinking> tags. Your answer to the user does not need tags. Try to always plan your next steps using the <thinking> tags, unless the query is very straightforward. Make sure to exhaust all your tools and delegate capabilities before providing your final answer. If the query is complicated, multiple thoughts are allowed.\n
"""

    _THINKING_PATTERN = re.compile(r"<thinking>(.*?)</thinking>", re.DOTALL)
    _DELEGATE_PATTERN = re.compile(r"<delegate>(.*?)</delegate>", re.DOTALL)

    #########
    # Setup #
    #########

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
        delegates: list["BedrockAgent"] | None = None,
        system_message: str | None = None,
        event_handlers: EventHandlers | dict[str, HandlerType] | None = None,
    ):
        """Initialize the BedrockAgent object.

        :param identifier: An identifier for the agent
        :param model: A Bedrock LLM model object
        :param description: A description of the agent, used in the agent's representation towards the user or other agents
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

        # Make sure delegates are marked as delegates
        for delegate in self.delegates:
            delegate._is_delegate = True

        # Set event handlers
        self._set_event_handlers(event_handlers)

        # Build and set the system message
        self._build_system_message()

        # Register tools with the model if supported
        self._register_tools()

        # Set is_delegate flag
        self._is_delegate = False

    def _build_system_message(self) -> str:
        """Build the system message with delegation options based on available delegates.

        :return: Complete system message including delegation options
        """
        system_message = self._BASE_SYSTEM_MESSAGE

        # Add description if available
        if self.description:
            system_message += f"This describes who you are: {self.description}\n\n"

        # Add delegation instructions if delegates are available
        if self.delegates:
            delegate_info = "You can delegate to other agents using <delegate>agent_name:query</delegate> tags. Be sure to include all the necessary information in the query tag, and that the query is in natural language. For example: <delegate>SomeSpecialistAgent:Can you do this specialized task?</delegate>.\n\n"

            # Get structured delegate information using a simplified version of the get_representation format
            delegate_info += "These are the delegate agents available to you, along with the tools that they have available. Note that you need to go through the delegate to access the tools:\n\n"
            for delegate in self.delegates:
                delegate_info += f"{delegate.get_representation()}"

            system_message += delegate_info

        # Append user system message if available
        if self._user_system_message:
            system_message += f"{self._user_system_message}"

        logger.debug(f"System message for {self.identifier}:\n\n{system_message}")

        self._system_message = system_message
        self.memory.set_system_message(system_message)

    def get_system_message(self) -> str:
        """Get the system message for the agent.

        :return: The system message for the agent
        """
        return self._system_message

    #######################
    # Event Handler Setup #
    #######################

    def _log_agent_event(self, message: str, log_type: AgentLogType) -> None:
        """Helper method to log agent events if logging is enabled.

        :param message: The message to log
        :param log_type: The type of log message
        """
        if self.log_agentic_response:
            self.log(message, log_type)

    def _set_event_handlers(
        self, event_handlers: EventHandlers | dict[str, HandlerType] | None = None
    ) -> None:
        """
        Set up the event handlers. If `log_agentic_response` is True, include default handlers to print agent events.
        Beyond that, use any user-provided handlers.

        :param event_handlers: Event handlers config for different agent events
        """

        # Initialize with default handlers if logging is enabled
        default_handlers = {}
        if self.log_agentic_response:
            default_handlers = {
                "on_start": [
                    lambda: self._log_agent_event(
                        f"Agent {self.identifier} started", AgentLogType.START
                    )
                ],
                "on_tool_use_start": [
                    lambda tool_name, parameters: self._log_agent_event(
                        f"Tool [{tool_name}] requested with parameters: {parameters}",
                        AgentLogType.TOOL_USE,
                    )
                ],
                "on_tool_use_stop": [
                    lambda tool_name, parameters, result: self._log_agent_event(
                        f"Tool [{tool_name}] executed with parameters: {parameters} -> {result}",
                        AgentLogType.TOOL_USE,
                    )
                ],
                "on_thinking": [
                    lambda text: self._log_agent_event(text, AgentLogType.THOUGHT)
                ],
                "on_answer": [
                    lambda text: self._log_agent_event(text, AgentLogType.ANSWER)
                ],
                "on_delegation_start": [
                    lambda delegate_name, query: self.log(
                        (
                            f'Initiating delegation to {delegate_name} with query: "{query}"'
                        ),
                        AgentLogType.DELEGATION,
                    )
                ],
                "on_delegation_stop": [
                    lambda delegate_name, query, answer, events: self.log(
                        f"Delegation to {delegate_name} concluded: {answer}",
                        AgentLogType.DELEGATION,
                    )
                ],
                "on_stop": [
                    lambda: self._log_agent_event(
                        f"Agent {self.identifier} stopped", AgentLogType.STOP
                    )
                ],
            }

        # Process user handlers if provided
        user_handlers = {}
        if event_handlers:

            # Convert dict to EventHandlers if needed
            if not isinstance(event_handlers, EventHandlers):
                try:
                    logger.debug(f"Converting dict to EventHandlers: {event_handlers}")
                    event_handlers = EventHandlers(**event_handlers)
                except Exception as e:
                    logger.warning(f"Error converting dict to EventHandlers: {e}")
                    raise ValueError(f"Invalid event handlers: {e}")

            # Extract non-None handlers
            user_handlers = {
                event_name: [handler] if not isinstance(handler, list) else handler
                for event_name, handler in event_handlers.model_dump(
                    exclude_none=True
                ).items()
            }

        # Merge default handlers with user handlers
        self.event_handlers = default_handlers.copy()
        for event_name, handlers in user_handlers.items():
            if event_name in self.event_handlers:
                self.event_handlers[event_name].extend(handlers)
            else:
                self.event_handlers[event_name] = handlers

    def _safe_event_handler(self, event_name: str, *args, **kwargs) -> None:
        """Safely dispatch event handlers, handling cases where the handler isn't assigned.

        :param event_name: The name of the event to invoke
        :param args: Positional arguments to pass to the event handler
        :param kwargs: Keyword arguments to pass to the event handler
        """
        handlers = self.event_handlers.get(event_name)
        if not handlers:
            return

        # Execute each handler safely
        for handler in handlers:
            if not callable(handler):
                logger.warning(f"Event handler for {event_name} is not callable")
                continue

            try:
                handler(*args, **kwargs)
            except Exception as e:
                logger.warning(
                    f"Error in {event_name} event handler (args: {args} - kwargs: {kwargs}): {e}",
                    exc_info=True,
                )

    ###############
    # Public APIs #
    ###############

    def run(
        self, prompt: str, max_iterations: int = 10, stream: bool = False
    ) -> AgentResponse:
        """Execute the agent with the given prompt.

        :param prompt: The initial prompt to feed to the LLM
        :param max_iterations: Maximum number of model-tool-model iterations
        :param stream: Whether to stream the response. Currently not implemented.
        :return: An AgentResponse object containing the answer and events
        :raises NotImplementedError: If stream=True, as streaming is not yet implemented
        """
        if stream:
            raise NotImplementedError(
                "Streaming is not yet implemented for BedrockAgent"
            )

        return self.invoke(prompt=prompt, max_iterations=max_iterations)

    def invoke(self, prompt: str, max_iterations: int = 10) -> AgentResponse:
        """Run the agent with the given prompt using the model's invoke method.

        :param prompt: The initial prompt to feed to the LLM
        :param max_iterations: Maximum number of model-tool-model iterations
        :return: An AgentResponse object containing the answer and events
        """
        # Reset memory and add prompt
        self.memory.add_message(role="user", content=prompt)

        # Initialize result containers
        all_events = []
        iterations = 0
        answer = None  # Initialize answer variable

        # Call on_start event handler
        if not self._is_delegate:
            self._safe_event_handler(event_name="on_start")
            all_events.append(
                AgentStartEvent(
                    agent_name=self.identifier, type=AgentEventTypes.START, content=None
                )
            )

        # Main agentic loop
        while iterations < max_iterations:

            # Get messages for the model
            prompt_obj = Messages(
                system=self.memory.get_system_message(),
                messages=self.memory.get_messages(),
            )

            # Process one iteration
            iteration_response = self._invoke_iteration(prompt_obj=prompt_obj)
            logger.debug(f"Iteration {iterations} events: {iteration_response}")

            # Collect events
            all_events.extend(iteration_response["events"])

            # First check for answer (highest priority)
            found_answer = False
            for event in iteration_response["events"]:
                if isinstance(event, AnswerEvent):
                    answer = event.content
                    # Found an answer, exit the loop
                    found_answer = True
                    break

            # Check for tool use or delegation to continue the loop
            found_action = False
            for event in iteration_response["events"]:
                if isinstance(event, ToolUseStartEvent) or isinstance(
                    event, DelegateStartEvent
                ):
                    found_action = True
                    break

            # If we found a tool use or delegate event, continue to next iteration
            if found_action:
                iterations += 1
                continue

            # ...otherwise, if we found an answer, we're done
            if found_answer:
                break

            # If we got here without an answer or tool/delegate use,
            # we're done because the model has nothing more to add
            logger.warning("Agentic loop interrupted: no events returned")
            break

        # Call on_stop event handler
        if not self._is_delegate:
            self._safe_event_handler(event_name="on_stop")
            all_events.append(
                AgentStopEvent(
                    agent_name=self.identifier, type=AgentEventTypes.STOP, content=None
                )
            )

        # Return results
        return AgentResponse(
            answer=answer,
            events=all_events,
        )

    ######################
    # Delegate Management #
    ######################

    def _execute_delegate(
        self, event: DelegateStartEvent, stream: bool = False
    ) -> tuple[str, list[AgentEvent]]:
        """Execute a delegate agent with the given query.

        :param event: DelegateEvent object containing the delegate name and query
        :return: Tuple of (delegate_answer, delegate_events)
        """
        # Extract delegate name and query from the event
        delegate_name = event.content.agent_name
        query = event.content.query

        # Call event handler before execution
        self._safe_event_handler(
            event_name="on_delegation_start",
            delegate_name=delegate_name,
            query=query,
        )

        # Find the delegate by name
        delegate = next(
            (d for d in self.delegates if d.identifier == delegate_name), None
        )

        if not delegate:
            error_msg = f"Delegate agent '{delegate_name}' not found. Available delegates: {[d.identifier for d in self.delegates]}"
            logger.warning(error_msg)
            self.memory.add_message(role="user", content=error_msg)
            return error_msg, []

        try:
            # Ensure the delegate registers its tools before execution
            if hasattr(delegate, "_register_tools"):
                delegate._register_tools()
                logger.debug(f"Re-registered tools for delegate {delegate_name}")

            # Execute delegate
            delegate_result = delegate.run(prompt=query, stream=stream)

            # Extract answer and events
            answer = delegate_result.answer
            delegate_events = delegate_result.events

            # Call event handler after execution with the result
            self._safe_event_handler(
                event_name="on_delegation_stop",
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

            return answer, delegate_events
        except Exception as e:
            error_result = self._handle_delegate_error(delegate_name, query, str(e))
            return error_result, []

    def _handle_delegate_error(
        self, delegate_name: str, query: str, error_msg: str
    ) -> str:
        """Handle an error that occurred during delegate execution.

        :param delegate_name: Name of the delegate that failed
        :param query: Query that was used to delegate
        :param error_msg: Error message from the delegate execution
        :return: Error message to be returned to the user
        """
        error_msg = f"Error delegating to {delegate_name}: {error_msg}. Available delegates: {self.delegates}."
        logger.error(error_msg)
        self.memory.add_message(role="user", content=error_msg)
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

        # Check if tools have a description
        for tool in self.tools:
            if not tool.get_tool_description():
                raise ValueError(f"Tool {tool.get_tool_name()} has no description")

        # Convert BaseTool objects to BedrockTool format and set on model
        bedrock_tools = [
            BedrockTool(**tool.model_dump_bedrock_converse()) for tool in self.tools
        ]
        self.model.toolConfig = BedrockToolConfig(tools=bedrock_tools)

        # Log the registered tools
        tool_names = [tool.get_tool_name() for tool in self.tools]
        logger.debug(
            f"Agent {self.identifier}: Registered {len(bedrock_tools)} tools with Bedrock model: {tool_names}"
        )

    def _execute_tool(self, tool_use_item: ToolUseContent) -> dict:
        """Execute a tool with the given parameters.

        :param tool_use_id: ID of the tool use
        :param tool_name: Name of the tool to call
        :param tool_parameters: Parameters for the tool
        :return: Dictionary with event and formatted result
        """

        # Extract commonly used values
        tool_name = tool_use_item.content.name
        tool_params = tool_use_item.content.input
        tool_use_id = tool_use_item.content.toolUseId

        logger.debug(f" Trying to execute tool: {tool_name} with params: {tool_params}")

        # Call event handler before execution
        self._safe_event_handler(
            event_name="on_tool_use_start",
            tool_name=tool_name,
            parameters=tool_params,
        )

        # Set start event
        tool_start_event = ToolUseStartEvent(
            agent_name=self.identifier,
            type=AgentEventTypes.TOOL_USE_START,
            content=ToolUseData(tool_name=tool_name, parameters=tool_params),
        )

        # Add tool use request to memory first
        self.memory.add_message(
            role="assistant",
            content=tool_use_item,
        )

        # Find the tool
        tool = next(
            (tool for tool in self.tools if tool.get_tool_name() == tool_name), None
        )
        if not tool:
            error_msg = f"[Tool Error: {tool_name}] Tool not found. Available tools: {[t.get_tool_name() for t in self.tools]}"
            self.memory.add_message(role="user", content=error_msg)
            return {"error": error_msg}

        try:
            logger.debug(
                f"🔨 Executing tool: [{tool_use_id}] {tool_name} with params: {tool_params}"
            )
            # Execute the tool
            tool_result = tool.run(environment=self.environment, **tool_params)

            # If the result is not a dict or a string, convert it to a string
            if not isinstance(tool_result, dict) and not isinstance(tool_result, str):
                tool_result = str(tool_result)

            # Call event handler
            self._safe_event_handler(
                event_name="on_tool_use_stop",
                tool_name=tool_name,
                parameters=tool_params,
                result=tool_result,
            )

            # Create formatted result for logs
            formatted_result = (
                f"[Tool used] {tool_name}({tool_params}) -> {tool_result}"
            )

            # Store structured tool data as an event
            tool_stop_event = ToolUseStopEvent(
                agent_name=self.identifier,
                type=AgentEventTypes.TOOL_USE_STOP,
                content=ToolUseData(
                    tool_name=tool_name,
                    parameters=tool_params,
                    result=tool_result,
                ),
            )

            # Then add the tool result
            self.memory.add_message(
                role="user",
                content=ToolResultContent(
                    type="toolResult",
                    content=ToolResultBlock(
                        content=[
                            (
                                ToolResultContentBlockJson(json=tool_result)
                                if isinstance(tool_result, dict)
                                else ToolResultContentBlockText(text=tool_result)
                            ),
                        ],
                        toolUseId=tool_use_id,
                        status="success",
                    ),
                ),
            )

            return {
                "formatted_result": formatted_result,
                "events": [tool_start_event, tool_stop_event],
            }
        except Exception as e:
            error_msg = (
                f"[Tool Error: {tool_name}] Error with parameters {tool_params}: {e}"
            )
            logger.error(error_msg)
            self.memory.add_message(role="user", content=error_msg)
            return {"error": error_msg}

    ######################
    # Content Processing #
    ######################

    def _process_content(self, content: list[Content]) -> list[AgentEvent]:
        """Process the content to extract thinking, answer, and delegate parts in chronological order.
        Any text not inside thinking or delegate tags will be treated as answer text.

        :param content: The content to process
        :return: A list of events
        """

        # Get the text content
        text_content = [block for block in content if isinstance(block, TextContent)]
        content = "".join([block.text for block in text_content])

        # Initialize events list
        events: list[AgentEvent] = []

        # Find all thinking and delegate tags
        thinking_matches = list(self._THINKING_PATTERN.finditer(content))
        delegate_matches = list(self._DELEGATE_PATTERN.finditer(content))

        # Create a list of all tagged sections with their start/end positions
        tagged_sections = []

        for match in thinking_matches:
            thought = match.group(1).strip()
            event = ThinkingEvent(
                agent_name=self.identifier,
                type=AgentEventTypes.THINKING,
                content=thought,
            )
            events.append(event)
            self._safe_event_handler(event_name="on_thinking", text=thought)
            tagged_sections.append((match.start(), match.end()))

        for match in delegate_matches:
            delegate_content = match.group(1).strip()
            try:
                agent_name, query = delegate_content.split(":", 1)
                agent_name = agent_name.strip()
                query = query.strip()
            except ValueError:
                agent_name, query = None, delegate_content

            event = DelegateStartEvent(
                agent_name=self.identifier,
                type=AgentEventTypes.DELEGATION_START,
                content=DelegateData(agent_name=agent_name, query=query),
            )
            events.append(event)
            tagged_sections.append((match.start(), match.end()))

        # Sort tagged sections by start position
        tagged_sections.sort()

        # Extract untagged text as answer
        answer_parts = []
        last_end = 0

        for start, end in tagged_sections:
            # Add text between last tag and current tag
            if start > last_end:
                untagged_text = content[last_end:start].strip()
                if untagged_text:
                    answer_parts.append(untagged_text)
            last_end = end

        # Add text after the last tag
        if last_end < len(content):
            untagged_text = content[last_end:].strip()
            if untagged_text:
                answer_parts.append(untagged_text)

        # If there were no tags at all, the entire content is the answer
        if not tagged_sections and content.strip():
            answer_parts = [content.strip()]

        # Join all answer parts and create an answer event if there's anything
        if answer_parts:
            answer_text = " ".join(answer_parts)
            if answer_text.strip():
                event = AnswerEvent(
                    agent_name=self.identifier,
                    type=AgentEventTypes.ANSWER,
                    content=answer_text,
                )
                events.append(event)
                self._safe_event_handler(event_name="on_answer", text=answer_text)

        return events

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
            f"Agent {self.identifier} is invoking the model:\n\n"
            f"\t[toolConfig]:\n\n{(pformat(self.model.toolConfig.model_dump(), indent=4) if self.model.toolConfig else 'No tools available')}\n"
            f"\t[prompt]:\n\n{pformat(prompt_obj.model_dump(), indent=4)}"
        )

        # Initialize event list for this iteration
        self._current_iteration_events = []

        try:
            response = self.model.invoke(prompt=prompt_obj)
            logger.debug(f"Agent got response:\n{pformat(response)}")

            # Initialize variables
            tool_use = []
            content = []

            # Extract message content
            if "output" in response and "message" in response["output"]:
                message = response["output"]["message"]

                # Process content blocks
                if "content" in message and isinstance(message["content"], list):

                    for item in message["content"]:

                        # Extract text content
                        if "text" in item:
                            logger.debug(f"Text found: {item}")
                            content.append(TextContent(text=item["text"]))

                        # Extract tool use information
                        if "toolUse" in item:
                            logger.debug(f"Tool use found: {item}")

                            # Add tool use to content
                            tool_use.append(
                                ToolUseContent(
                                    content=ToolUseBlock(
                                        toolUseId=item["toolUse"]["toolUseId"],
                                        name=item["toolUse"]["name"],
                                        input=item["toolUse"]["input"],
                                    )
                                )
                            )

            elif isinstance(response, str):
                content.append(TextContent(text=response))

            # Process content for thinking/delegate tags
            if content:

                # Process the content for thinking/delegate tags and trigger event handlers
                try:
                    events = self._process_content(content)
                except Exception as e:
                    logger.error(f"Error processing content: {e}")
                    events = []

                # Add response to memory
                for block in content:
                    self.memory.add_message(role="assistant", content=block)

                # If a delegate event is found, we need to execute the delegate
                for i, event in enumerate(events):
                    if isinstance(event, DelegateStartEvent):
                        delegate_answer, delegate_events = self._execute_delegate(event)
                        delegate_stop_event = DelegateStopEvent(
                            agent_name=self.identifier,
                            type=AgentEventTypes.DELEGATION_STOP,
                            content=DelegateData(
                                agent_name=event.content.agent_name,
                                query=event.content.query,
                                answer=delegate_answer,
                                events=delegate_events,
                            ),
                        )
                        # Insert the stop event right after the start event
                        events.insert(i + 1, delegate_stop_event)

                # Add events to current iteration
                self._current_iteration_events.extend(events)

            # Process tool call if found
            if tool_use:
                for tool_use_item in tool_use:
                    tool_result = self._execute_tool(tool_use_item)
                if "events" in tool_result:
                    self._current_iteration_events.extend(tool_result["events"])
                elif "error" in tool_result:
                    logger.error(tool_result["error"])

        except Exception as e:
            error_message = f"Error during model invocation: {str(e)}"
            logger.error(error_message)

            # Add error message to memory so the agent knows something went wrong
            self.memory.add_message(
                role="user",
                content=f"[SYSTEM ERROR] {error_message}. Please try a different approach.",
            )

            # Create a thinking event to indicate the error
            error_event = ThinkingEvent(
                agent_name=self.identifier,
                type=AgentEventTypes.THINKING,
                content=f"Encountered an error: {error_message}",
            )
            self._current_iteration_events.append(error_event)
            self._safe_event_handler(event_name="on_thinking", text=error_event.content)

        # Store events and clean up
        iteration_events = self._current_iteration_events
        delattr(self, "_current_iteration_events")

        return {
            "events": iteration_events,
        }

    def _flush_memory(self):
        """Clear or reset the agent's memory context.

        Overrides the BaseAgent implementation to ensure message list is cleared.
        """
        # Get the current system message (if any)
        system_message = self.memory.get_system_message()

        # Clear all messages
        self.memory.get_messages().clear()

        # Set the system message (either preserved or new)
        if system_message:
            self.memory.set_system_message(system_message)
        elif self._system_message:
            self.memory.set_system_message(self._system_message)

        # If we have a prefill, add it
        if self.prefill:
            self.memory.add_message(role="assistant", content=self.prefill)


if __name__ == "__main__":

    from fence.tools.base import BaseTool, tool
    from fence.utils.logger import setup_logging

    MODEL = Claude35Sonnet(region="us-east-1")
    MODEL2 = NovaPro(
        region="us-east-1",
    )

    setup_logging(log_level="info", are_you_serious=False)

    # Create a test tool
    @tool(description="Returns the age of the user")
    def age_lookup_tool() -> str:
        """Test tool"""
        return {"result": "You are 25 years old."}

    @tool(description="Returns the name of the current user")
    def name_lookup_tool() -> str:
        """Test tool"""
        return "Hello, Max! Your name is Max."

    @tool(description="Checks eligibility for a loan")
    def check_eligibility(name: str, age: int) -> str:
        """Check eligibility for a loan"""
        if age < 18:
            return f"Hello, {name}! You are {age} years old. You are not eligible for a loan."
        else:
            return (
                f"Hello, {name}! You are {age} years old. You are eligible for a loan."
            )

    eligibility_agent = BedrockAgent(
        identifier="EligibilityAgent",
        model=MODEL,
        description="An specialist agent that has various capabilities and tools to check eligibility for a loan. Only requires an age and name to check eligibility.",
        tools=[check_eligibility],
        log_agentic_response=True,
    )

    bank_agent = BedrockAgent(
        identifier="BankAgent",
        model=MODEL,
        tools=[age_lookup_tool, name_lookup_tool],
        # delegates=[eligibility_agent],
        description="A helpful assistant that can check eligibility for a loan.",
        log_agentic_response=True,
    )

    # Invoke demo
    response = None
    while response != "exit":
        user_query = input("🤺 ")
        response = bank_agent.run(
            user_query,
            max_iterations=4,
        )
        print(response.answer)
    # pprint(response.answer)
    # pprint(response.events)

    # print(bank_agent._system_message)
    # print(bank_agent.get_representation())
