"""
Base streaming handler for processing text chunks with tagged sections.
"""

import json
import logging
import re
from typing import Any, Callable, Generator

from fence.agents.bedrock.models import (
    AgentEvent,
    AnswerEvent,
    DelegateData,
    DelegateEvent,
    ThinkingEvent,
    ToolUseData,
    ToolUseEvent,
)

logger = logging.getLogger(__name__)


class StreamHandler:
    """
    Process text chunks with tagged sections and emit events for each section.

    This class handles streaming text that contains tagged sections in the format
    <tag>content</tag> and processes them into discrete events. It's designed to
    work with partial chunks of text that may arrive over time, making it suitable
    for real-time streaming applications.

    The handler maintains an internal buffer and state to track partially received
    tags and content. When a complete section is identified (content between opening
    and closing tags), it emits an event via the provided callback function.

    Example:
        >>> handler = StreamHandler()
        >>> handler.process_chunk("<thinking>Let me consider this")
        >>> handler.process_chunk(" question carefully</thinking>")
        EVENT [thinking]: Let me consider this question carefully

    The handler can process multiple tag types in the same stream and handles
    nested content appropriately. It's resilient to receiving partial tags
    split across multiple chunks.
    """

    def __init__(self, event_callback: Callable[[str, str], None] | None = None):
        """
        Initialize the stream handler.

        :param event_callback: Function to call for each event (tag content), defaults to print
        """
        self.buffer = ""
        self.current_mode: str | None = None
        self.event_callback = event_callback or self._default_event_callback

        # Initialize collection for events in chronological order
        self.events: list[dict[str, Any]] = []

        # Flag to track if we're at the start of a tag mode (to handle leading whitespace)
        self.at_tag_start = False

    def process_chunk(self, chunk: str) -> None:
        """
        Process a single text chunk.

        :param chunk: Text chunk to process
        """
        self.buffer += chunk

        # Continue processing until no more changes
        processing = True
        while processing:
            processing = False

            # If not in a mode, look for any opening tag <tag>
            if self.current_mode is None:

                # Check for a complete opening tag pattern: <tag>
                opening_match = self._find_opening_tag(self.buffer)
                if opening_match:
                    tag_name, start_pos, end_pos = opening_match

                    # Extract content after the tag
                    self.buffer = self.buffer[end_pos:]
                    self.current_mode = tag_name

                    # Strip any leading whitespace when entering a new tag mode
                    self.buffer = self.buffer.lstrip()

                    # Set flag to indicate we're at the start of a tag
                    self.at_tag_start = True
                    processing = True
                    continue

            # If in a mode, look for corresponding closing tag
            elif self.current_mode:

                closing_tag = f"</{self.current_mode}>"
                if closing_tag in self.buffer:

                    # Found complete closing tag
                    parts = self.buffer.split(closing_tag, 1)
                    content = parts[0]

                    # Emit the content before the closing tag
                    if content:
                        # Strip trailing whitespace from the last content chunk
                        content = content.rstrip()
                        if (
                            content
                        ):  # Only emit and store if content is not empty after stripping
                            self._emit_content(content)
                            self._store_event(content)

                    # Reset for next section
                    self.current_mode = None
                    self.at_tag_start = False
                    self.buffer = parts[1]
                    processing = True
                    continue

                # Check for partial closing tag at the end
                elif self._has_partial_closing_tag(self.buffer, self.current_mode):

                    # Wait for more chunks
                    break

                # No complete or partial closing tag, emit buffer content and clear it
                elif self.buffer and self.current_mode != "delegate":

                    # If we're at the start of a tag, make sure all leading whitespace is removed
                    if self.at_tag_start:
                        self.buffer = self.buffer.lstrip()
                        self.at_tag_start = False

                    # Strip trailing whitespace from the last content chunk, but only if it's not just whitespace
                    if self.buffer.strip():
                        content = self.buffer.rstrip()
                    else:
                        # If buffer is only whitespace, preserve it
                        content = self.buffer

                    if content:
                        # Only emit and store if content is not empty after stripping
                        self._emit_content(content)
                        self._store_event(content)
                    self.buffer = ""
                    break

    def _emit_content(self, content: str) -> None:
        """Emit content through the event callback.

        :param content: Content to emit
        """
        if self.current_mode:
            self.event_callback(self.current_mode, content)

    def _store_event(self, content: str) -> None:
        """Store content as an event with a type.

        :param content: Content to store
        """
        if self.current_mode:
            event = {
                "type": self.current_mode,
                "content": content,
            }
            self.events.append(event)

    def _find_opening_tag(self, text: str) -> tuple[str, int, int] | None:
        """
        Find the first complete opening tag in the text.

        :param text: Text to search for opening tags
        :return: Tuple of (tag_name, start_pos, end_pos) or None if not found
        """
        # Match any opening tag pattern <tag>
        match = re.search(r"<([a-zA-Z0-9_]+)>", text)
        if match:
            tag_name = match.group(1)
            return (tag_name, match.start(), match.end())
        return None

    def _has_partial_closing_tag(self, text: str, tag_name: str) -> bool:
        """
        Check if text ends with a partial closing tag for the given tag name.

        :param text: Text to check
        :param tag_name: Name of the tag to check for
        :return: True if text ends with a partial closing tag
        """
        closing_tag = f"</{tag_name}>"
        return any(text.endswith(closing_tag[:i]) for i in range(1, len(closing_tag)))

    def _default_event_callback(self, event_type: str, content: str) -> None:
        """
        Default callback for events.

        :param event_type: Type of event (tag name)
        :param content: Content to send with the event
        """
        print(f"EVENT [{event_type}]: {content}")

    def reset(self) -> None:
        """Reset the stream handler."""
        self.buffer = ""
        self.current_mode = None
        self.at_tag_start = False
        self.events = []


class ConverseStreamHandler:
    """
    Process streaming chunks from the Converse API and emit AgentEvent objects.

    This handler processes chunks from the Amazon Bedrock Converse API streaming format,
    which consist of JSON objects with different event types (messageStart, contentBlockDelta,
    messageStop, etc.). It extracts the text and tool use data from these chunks and emits
    proper AgentEvent objects as defined in fence.agents.bedrock.models.

    Example:
        >>> handler = ConverseStreamHandler()
        >>> for event in handler.process_chunks(converse_response):
        >>>     print(f"Received event: {event}")
    """

    def __init__(self):
        """
        Initialize the ConverseStreamHandler.
        """
        # Create stream handler with our own callback for text events
        self.events: list[AgentEvent] = []
        self.text_events = []  # Temporary storage for events from text_stream_handler
        self.text_stream_handler = StreamHandler(event_callback=self._handle_text_event)
        self.current_mode = None
        self.current_role = None
        self.current_buffer = None

        # Delegate event buffering
        self.is_buffering_delegate = False
        self.delegate_buffer = None

    def _handle_text_event(self, event_type: str, content: str) -> None:
        """
        Handle text events from the stream handler.

        :param event_type: The type of event (thinking, answer, delegate)
        :param content: The content of the event
        """
        # Convert text event to appropriate AgentEvent based on type and store
        if event_type == "thinking":
            self.text_events.append(ThinkingEvent(content=content))
        elif event_type == "answer":
            self.text_events.append(AnswerEvent(content=content))
        elif event_type == "delegate":
            # For delegate events, attempt to parse into agent_name and query
            try:
                # Split on the last colon to better handle agent names with colons
                if ":" in content:
                    agent_name, query = content.split(":", 1)
                    # Only strip whitespace from ends, preserve internal spaces
                    agent_name = agent_name.strip()
                    query = query.strip()

                    if agent_name and query:
                        # Complete delegate event with both name and query
                        self.text_events.append(
                            DelegateEvent(
                                content=DelegateData(agent_name=agent_name, query=query)
                            )
                        )
                    else:
                        raise ValueError("Invalid delegate event format")
                else:
                    # Not in the correct format yet, keep buffering
                    self.is_buffering_delegate = True
                    self.delegate_buffer = content
            except ValueError:
                # Not in the correct format yet, keep buffering
                self.is_buffering_delegate = True
                self.delegate_buffer = content

    def process_chunk(self, chunk: dict) -> Generator[AgentEvent, None, None]:
        """
        Process a single chunk from the Converse API and yield any events.

        :param chunk: JSON object from the Converse API
        :yield: AgentEvent objects produced from this chunk
        """
        # Clear temporary event storage
        self.text_events = []

        if "messageStart" in chunk:
            self.current_role = chunk["messageStart"]["role"]
            self.current_mode = "text"
            self.buffer = ""
            self.text_stream_handler.reset()
            self.is_buffering_delegate = False
            self.delegate_buffer = None

        elif "contentBlockDelta" in chunk:
            if self.current_mode == "text":

                # This will add events to self.text_events via _handle_text_event
                self.text_stream_handler.process_chunk(
                    chunk["contentBlockDelta"]["delta"]["text"]
                )

                # Only yield events if we're not buffering a delegate event
                if not self.is_buffering_delegate:
                    for event in self.text_events:
                        self.events.append(event)
                        yield event

            elif self.current_mode == "tool_use":
                if "input" in chunk["contentBlockDelta"]["delta"]["toolUse"]:
                    chunk["contentBlockDelta"]["delta"]["toolUse"]["input"] = (
                        json.loads(
                            chunk["contentBlockDelta"]["delta"]["toolUse"]["input"]
                        )
                    )
                self.current_buffer |= chunk["contentBlockDelta"]["delta"]["toolUse"]

        elif "contentBlockStop" in chunk:
            if self.current_mode == "tool_use":
                tool_use = self.current_buffer
                self.current_mode = None
                self.current_buffer = None

                # Create and yield tool use event
                event = ToolUseEvent(
                    content=ToolUseData(
                        name=tool_use["name"], parameters=tool_use["input"], result=None
                    )
                )
                self.events.append(event)
                yield event

        elif "contentBlockStart" in chunk:
            if chunk["contentBlockStart"].get("start", {}).get("toolUse"):
                self.current_mode = "tool_use"
                self.current_buffer = chunk["contentBlockStart"]["start"]["toolUse"]

    def reset(self) -> None:
        """Reset the handler state."""
        self.current_mode = None
        self.current_buffer = None
        self.text_stream_handler.reset()
        self.text_events = []
        self.is_buffering_delegate = False
        self.delegate_buffer = None


if __name__ == "__main__":
    # Import here for the example
    from fence.agents.bedrock.models import (
        AgentEvent,
        AnswerEvent,
        DelegateData,
        DelegateEvent,
        ThinkingEvent,
        ToolUseData,
        ToolUseEvent,
    )

    ######################
    # Test StreamHandler #
    ######################

    handler = StreamHandler()

    dummy_chunks = [
        "<th",
        "inking>Hello,",
        " world",
        "</",
        "thin",
        "king><a",
        "nswer>Sup",
        "?<",
        "/answer",
        ">",
        "<deleg",
        "ate>agent1:",
        "query",
        "</delegate>",
    ]

    for chunk in dummy_chunks:
        handler.process_chunk(chunk)

    print("\nFinal events:", len(handler.events))
    for i, event in enumerate(handler.events):
        print(event)

    ##############################
    # Test ConverseStreamHandler #
    ##############################

    handler = ConverseStreamHandler()

    dummy_chunks = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"delta": {"text": "<"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": "thinking"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": ">"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": " As"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": " an"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": " AI"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": ","}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": " I"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": " don"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": "'"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": "t"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": " have"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": " personal"}, "contentBlockIndex": 0}},
        {
            "contentBlockDelta": {
                "delta": {"text": " experiences"},
                "contentBlockIndex": 0,
            }
        },
        {"contentBlockDelta": {"delta": {"text": " or"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": " emotions"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": ","}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": " so"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": " I"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": " can"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": "'"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": "t"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": " feel"}, "contentBlockIndex": 0}},
        {
            "contentBlockDelta": {
                "delta": {"text": " happiness"},
                "contentBlockIndex": 0,
            }
        },
        {"contentBlockDelta": {"delta": {"text": ","}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": " sadness"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": ","}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": " or"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": " any"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": " other"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": " emotions"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": "."}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": " However"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": ","}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": " I"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": "'"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": "m"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": " fully"}, "contentBlockIndex": 0}},
        {
            "contentBlockDelta": {
                "delta": {"text": " operational"},
                "contentBlockIndex": 0,
            }
        },
        {"contentBlockDelta": {"delta": {"text": " and"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": " ready"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": " to"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": " assist"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": " with"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": " any"}, "contentBlockIndex": 0}},
        {
            "contentBlockDelta": {
                "delta": {"text": " questions"},
                "contentBlockIndex": 0,
            }
        },
        {"contentBlockDelta": {"delta": {"text": " or"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": " tasks"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": " you"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": " have"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": "."}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": " </"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": "thinking"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": ">\n\n"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": "<"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": "answer"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": ">"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": " Hello"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": "!"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": " I"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": "'"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": "m"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": " here"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": " and"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": " ready"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": " to"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": " assist"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": " you"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": " with"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": " any"}, "contentBlockIndex": 0}},
        {
            "contentBlockDelta": {
                "delta": {"text": " questions"},
                "contentBlockIndex": 0,
            }
        },
        {"contentBlockDelta": {"delta": {"text": " or"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": " tasks"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": " you"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": " have"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": "."}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": " How"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": " can"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": " I"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": " help"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": " you"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": " today"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": "?"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": " </"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": "answer"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": ">"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": ""}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": "<delegate"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": ">"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": "E"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": "lig"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": "ibility"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": " Agent"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": ":"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": "Check"}, "contentBlockIndex": 0}},
        {
            "contentBlockDelta": {
                "delta": {"text": " eligibility"},
                "contentBlockIndex": 0,
            }
        },
        {"contentBlockDelta": {"delta": {"text": " for"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": " a"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": " loan"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": " for"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": " Max"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": ","}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": " who"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": " is"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": " "}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": "2"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": "5"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": " years"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": " old"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": ".</"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": "delegate"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": ">"}, "contentBlockIndex": 0}},
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {
            "contentBlockStart": {
                "start": {
                    "toolUse": {
                        "toolUseId": "tooluse_Tn8F-uYLSlCZq23tb3HaUA",
                        "name": "AgeLookupTool",
                    }
                },
                "contentBlockIndex": 1,
            }
        },
        {
            "contentBlockDelta": {
                "delta": {"toolUse": {"input": '{"name":"Max"}'}},
                "contentBlockIndex": 1,
            }
        },
        {"contentBlockStop": {"contentBlockIndex": 1}},
        {"messageStop": {"stopReason": "tool_use"}},
        {
            "metadata": {
                "usage": {"inputTokens": 469, "outputTokens": 69, "totalTokens": 538},
                "metrics": {"latencyMs": 825},
            }
        },
    ]

    for chunk in dummy_chunks:
        for event in handler.process_chunk(chunk):
            print(event)

    print("\nFinal events:", len(handler.events))
    for i, event in enumerate(handler.events):
        print(f"Event {i}: {event}")
