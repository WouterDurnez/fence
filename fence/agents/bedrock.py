"""
Bedrock agent class that uses native tool calling and streaming capabilities.
"""

import json
import logging
import re
import sys
from typing import Any, Callable, Iterator

from fence.agents.base import AgentLogType, BaseAgent
from fence.memory.base import BaseMemory, FleetingMemory
from fence.models.base import LLM
from fence.models.bedrock.base import BedrockTool, BedrockToolConfig
from fence.templates.models import Messages
from fence.tools.base import BaseTool

logger = logging.getLogger(__name__)


SYSTEM_MESSAGE = """
You are a helpful assistant that can provide weather information and perform temperature conversions. Always start by planning your response in <thinking>...</thinking> tags. Return your final response in <answer>...</answer> tags.

IMPORTANT INSTRUCTION: You must be extremely direct and concise. Never acknowledge or thank users for tool results. After a tool returns a result, immediately continue to the next logical step without any transition phrases like "Certainly", "Now", or "Thank you". You are allowed to think, using <thinking>...</thinking> tags. If the next step is to use another tool, immediately call that tool. If all tools have been used, immediately provide your final answer in the requested format without any introduction.
"""


class BedrockAgent(BaseAgent):
    """
    Bedrock agent that uses native tool calling and streaming capabilities.
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
        callbacks: dict[str, Callable] | None = None,
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
        :param callbacks: A dictionary of callbacks for different agent events:
                         - 'on_action': Called when the agent uses a tool
                         - 'on_observation': Called when a tool returns a result
                         - 'on_answer': Called when the agent provides text answer chunks
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

        # Save and set the system message
        if system_message:
            # Combine with the default system message
            self._system_message = f"{SYSTEM_MESSAGE}\n\n{system_message}"
        else:
            # Use just the default system message
            self._system_message = SYSTEM_MESSAGE

        # Set the system message in memory
        self.memory.set_system_message(self._system_message)

        self.tools = tools or []

        # Set up callbacks with defaults
        self.callbacks = {
            "on_action": self._default_on_action,
            "on_observation": self._default_on_observation,
            "on_answer": self._default_on_answer,
        }

        # Update with user-provided callbacks
        if callbacks:
            self.callbacks.update(callbacks)

        # Register tools with the model if supported
        self._register_tools()

    #############
    # Callbacks #
    #############

    def _default_on_action(self, tool_name: str, parameters: dict) -> None:
        """Default callback for tool actions.

        :param tool_name: The name of the tool being called
        :param parameters: The parameters passed to the tool
        """
        self.log(
            f"Using tool [{tool_name}] with parameters: {parameters}",
            AgentLogType.ACTION,
        )

    def _default_on_observation(self, tool_name: str, result: Any) -> None:
        """Default callback for tool observations.

        :param tool_name: The name of the tool that was called
        :param result: The result returned by the tool
        """
        self.log(
            f"Tool result [{tool_name}]: {result}",
            AgentLogType.OBSERVATION,
        )

    def _default_on_answer(self, text: str) -> None:
        """Default callback for agent answers.

        :param text: The text chunk produced by the agent
        """
        self.log(text, AgentLogType.ANSWER)

    def _safe_callback(self, callback_name: str, *args, **kwargs) -> None:
        """
        Safely dispatch a callback, handling cases where the callback isn't assigned.

        :param callback_name: The name of the callback to invoke
        :param args: Positional arguments to pass to the callback
        :param kwargs: Keyword arguments to pass to the callback
        """
        callback = self.callbacks.get(callback_name)
        if callback and callable(callback):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Error in {callback_name} callback: {e}")
                # Continue execution even if callback fails

    ###########
    # Helpers #
    ###########

    def _register_tools(self):
        """
        Register tools with the Bedrock model if supported.
        """
        if not self.tools or not self.model:
            return

        # Check if the model has the toolConfig attribute
        if not hasattr(self.model, "toolConfig"):
            logger.warning(
                f"Model {self.model.__class__.__name__} does not support tool registration"
            )
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
        # Call the action callback
        self._safe_callback("on_action", tool_name, tool_parameters)
        tool_result_message = ""

        # Find and execute the matching tool
        for tool in self.tools:
            if tool.get_tool_name() == tool_name:
                try:
                    tool_result = tool.run(
                        environment=self.environment,
                        **tool_parameters,
                    )

                    # Call the observation callback
                    self._safe_callback("on_observation", tool_name, tool_result)

                    # Create tool message for logs
                    tool_result_message = (
                        f"[Tool Result] {tool_name}({tool_parameters}) -> {tool_result}"
                    )

                    # Add tool result to memory as user message
                    self.memory.add_message(
                        role="user",
                        content=f"[SYSTEM DIRECTIVE] The {tool_name} returned: {tool_result}. DO NOT ACKNOWLEDGE THIS MESSAGE. FIRST THINK, THEN PROCEED IMMEDIATELY to either: (1) Call the next required tool without any introduction or transition phrases, or (2) If all necessary tools have been used, provide your final answer. Think back to the original user prompt and use that to guide your response.",
                    )

                except Exception as e:
                    error_msg = f"[Tool Error: {tool_name}] {str(e)}"

                    # Log the error using the observation callback
                    self._safe_callback("on_observation", tool_name, f"Error: {str(e)}")

                    # Add error to memory
                    self.memory.add_message(
                        role="user",
                        content=f"Your tool call resulted in an error: {str(e)}. Please try a different approach.",
                    )

                    tool_result_message = error_msg

                break

        return tool_result_message

    def _process_content(self, content: str) -> tuple[list[str], list[str]]:
        """
        Process the content to extract thinking and answer parts.

        :param content: The content to process
        :return: A tuple of (thinking, answer)
        """
        thinking = []
        answer = []

        # Find all thinking and answer tags
        thinking_matches = self._THINKING_PATTERN.findall(content)
        answer_matches = self._ANSWER_PATTERN.findall(content)

        # Strip whitespace from matches
        thinking_matches = [match.strip() for match in thinking_matches]
        answer_matches = [match.strip() for match in answer_matches]

        # If no matches are found, return empty lists
        if not thinking_matches and not answer_matches:
            return [], []

        # Add the matches to the respective lists
        thinking.extend(thinking_matches)
        answer.extend(answer_matches)

        return thinking, answer

    #######
    # Run #
    #######

    def run(
        self, prompt: str, max_iterations: int = 10, stream: bool = False
    ) -> str | Iterator[str]:
        """
        Run the agent with the given prompt.

        :param prompt: The initial prompt to feed to the LLM
        :param max_iterations: Maximum number of model-tool-model iterations
        :param stream: Whether to stream the response or return the full text
        :return: The agent's response as a string or an iterator of response chunks if stream=True
        """
        # If streaming is requested, use the stream method directly
        if stream:
            return self.stream(prompt, max_iterations)

        response, thinking, answer = self.invoke(prompt, max_iterations)
        return response, thinking, answer

    def _invoke_iteration(self, prompt_obj: Messages) -> tuple[str, bool, list[str]]:
        """
        Process a single iteration of the agent's conversation with the model using invoke.

        :param prompt_obj: Messages object containing the conversation history
        :return: Tuple of (response_text, tool_used_flag, tool_result_messages)
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
        tool_result_messages = []

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
            self._safe_callback("on_answer", content)

        # Process tool call if found in the response
        if tool_used and tool_name and tool_parameters:
            # Use the centralized _process_tool_call method
            tool_result = self._process_tool_call(tool_name, tool_parameters)
            tool_result_messages.append(tool_result)
            return content, thinking, answer, True, tool_result_messages

        return content, thinking, answer, tool_used, tool_result_messages

    def invoke(self, prompt: str, max_iterations: int = 10) -> str:
        """
        Run the agent with the given prompt using the model's invoke method.

        :param prompt: The initial prompt to feed to the LLM
        :param max_iterations: Maximum number of model-tool-model iterations
        :return: The agent's response as a string
        """
        # Clear or reset the agent's memory context
        self._flush_memory()

        # Add the user's prompt to memory
        self.memory.add_message(role="user", content=prompt)

        # Initialize buffers for thinking and answer
        all_thinking = []
        all_answer = []
        full_response = []
        iterations = 0

        while iterations < max_iterations:

            # Get current messages and system
            messages = self.memory.get_messages()
            system = self.memory.get_system_message()

            # Prep messages for the model
            prompt_obj = Messages(system=system, messages=messages)

            # Process one iteration and get response and tool use flag
            (
                response,
                iteration_thinking,
                iteration_answer,
                tool_used,
                tool_result_messages,
            ) = self._invoke_iteration(prompt_obj)

            # Add thoughts and answer to the buffers
            all_thinking.extend(iteration_thinking)
            all_answer.extend(iteration_answer)

            # Add to the full response if we got a valid response
            if response:
                full_response.append(response)

            # Add tool result messages to the full response
            if tool_result_messages:
                full_response.extend(tool_result_messages)

            # If no tool was used, we're done
            if not tool_used:
                break

            # Increment iteration counter
            iterations += 1

            # If we've hit max iterations, exit
            if iterations >= max_iterations:
                logger.warning(
                    f"Reached maximum iterations ({max_iterations}). Stopping."
                )
                break

        # There should only be one answer
        answer = all_answer[0]

        return "\n\n".join(full_response), all_thinking, answer

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

        # Stream and process the response
        for chunk in self.model.stream(prompt=prompt_obj):
            # Handle different chunk types
            if "contentBlockDelta" in chunk:
                delta = chunk["contentBlockDelta"]["delta"]

                # Handle text deltas
                if "text" in delta:
                    text = delta["text"]
                    iteration_response += text

                    # Call the answer callback
                    self._safe_callback("on_answer", text)
                    yield text

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

                    # Process the tool call but don't yield the message
                    self._process_tool_call(
                        current_tool_name, current_tool_data["parameters"]
                    )

                    # Reset tool collection state
                    is_collecting_tool_data = False
                    current_tool_name = None
                    current_tool_data = {}
                    current_tool_buffer = ""

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

        # Return tool_used as a hidden value for the stream method
        if tool_used:
            # This is a special marker that will not be displayed but will be detected by the stream method
            yield "[[TOOL_USED]]"

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

    # Create tools
    weather_tool = get_weather
    convert_tool = convert_temperature
    print(f"Tool types: {type(weather_tool)}, {type(convert_tool)}")

    # Define a callback handler class to avoid using globals
    class CallbackHandler:
        """Callback handler for managing agent callbacks with internal state."""

        def __init__(self):
            """Initialize the callback handler with default state."""
            self.last_output_was_tool_result = False

        def on_action(self, tool_name, parameters):
            """Handle tool action events.

            :param tool_name: Name of the tool being called
            :param parameters: Parameters passed to the tool
            """
            print(f"\n\n🔧 TOOL CALLED: {tool_name} with {parameters}")

        def on_observation(self, tool_name, result):
            """Handle tool observation events.

            :param tool_name: Name of the tool that was called
            :param result: Result returned by the tool
            """
            print(f"📊 TOOL RESULT: {result}")
            # Set flag to indicate we just had a tool result
            self.last_output_was_tool_result = True

        def on_answer(self, text):
            """Handle agent response text.

            :param text: Text chunk produced by the agent
            """
            # Check if we need to add extra spacing after tool result
            if self.last_output_was_tool_result:
                # Add double newline after a tool result when text starts coming in
                sys.stdout.write("\n")
                # Reset the flag
                self.last_output_was_tool_result = False

            # Print text appropriately
            sys.stdout.write(text)
            sys.stdout.flush()

    # Create the callback handler instance
    callback_handler = CallbackHandler()

    # Example 3: Using BedrockAgent with multiple tools
    logger.info("\nExample 3: Using BedrockAgent with multiple tools:")
    agent = BedrockAgent(
        identifier="WeatherAssistant",
        model=Claude35Sonnet(),
        description="An assistant that can provide weather information and perform temperature conversions",
        tools=[weather_tool, convert_tool],
        memory=FleetingMemory(),
        log_agentic_response=False,  # Disable default logging since we're using callbacks
        system_message=SYSTEM_MESSAGE,
        callbacks={
            "on_action": callback_handler.on_action,
            "on_observation": callback_handler.on_observation,
            "on_answer": callback_handler.on_answer,
        },
    )

    # Run the agent with a prompt that requires multiple tool calls
    prompt = "What's the current weather in Tokyo and then convert the temperature to Celsius?"
    # prompt = "Hi, how are you?"
    print(f"\nUser question: {prompt}")

    # Use streaming mode
    for chunk in agent.run(prompt, stream=True):
        # Chunks are already being handled by the on_answer callback
        # This loop just ensures we process all chunks
        pass

    # Run the agent with streaming set to False
    logger.info("\nExample 4: Using BedrockAgent with streaming set to False")

    response, thinking, answer = agent.run(prompt, stream=False)
    print(f"Answer: {answer}")
    print(f"Thinking: {thinking}")
    # print(f"Response: {response}")
