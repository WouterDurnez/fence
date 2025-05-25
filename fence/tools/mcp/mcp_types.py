"""Type definitions for MCP integration."""

from contextlib import AbstractAsyncContextManager
from typing import TypeAlias

from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from mcp.client.streamable_http import GetSessionIdCallback
from mcp.shared.memory import MessageStream
from mcp.shared.message import SessionMessage

# _MessageStreamWithGetSessionIdCallback defines a type for streams used in HTTP streaming with session ID callbacks
# It consists of:
# 1. A receive stream for SessionMessages or Exceptions
# 2. A send stream for SessionMessages 
# 3. A GetSessionIdCallback function
_MessageStreamWithGetSessionIdCallback: TypeAlias = tuple[
    MemoryObjectReceiveStream[SessionMessage | Exception], 
    MemoryObjectSendStream[SessionMessage], 
    GetSessionIdCallback
]

# MCPTransport defines the interface for MCP transport implementations.
# It abstracts communication with an MCP server, hiding details of the underlying
# transport mechanism (WebSocket, stdio, HTTP streaming, etc.).
#
# It represents an async context manager that yields a tuple of read and write streams for MCP communication.
# When used with `async with`, it should establish the connection and yield the streams, then clean up
# when the context is exited.
#
# The read stream receives messages from the server (or exceptions if parsing fails),
# while the write stream sends messages to the server.
#
# Example implementation (simplified):
# ```python
# @contextlib.asynccontextmanager
# async def my_transport_implementation():
#     # Set up connection
#     read_stream_writer, read_stream = anyio.create_memory_object_stream(0)
#     write_stream, write_stream_reader = anyio.create_memory_object_stream(0)
#     
#     # Start background tasks to handle actual I/O
#     async with anyio.create_task_group() as tg:
#         tg.start_soon(reader_task, read_stream_writer)
#         tg.start_soon(writer_task, write_stream_reader)
#         
#         # Yield the streams to the caller
#         yield (read_stream, write_stream)
# ```
MCPTransport: TypeAlias = AbstractAsyncContextManager[MessageStream | _MessageStreamWithGetSessionIdCallback]
