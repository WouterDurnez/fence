"""
Base streaming handler for processing text chunks with tagged sections.
"""

import logging
import re
from typing import Any, Callable

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
                elif self.buffer:
                    # If we're at the start of a tag, make sure all leading whitespace is removed
                    if self.at_tag_start:
                        self.buffer = self.buffer.lstrip()
                        self.at_tag_start = False

                    # Strip trailing whitespace from the last content chunk
                    content = self.buffer.rstrip()
                    if (
                        content
                    ):  # Only emit and store if content is not empty after stripping
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
