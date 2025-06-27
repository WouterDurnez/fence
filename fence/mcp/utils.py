from typing import Any

from mcp import StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client


def create_stdio_transport(command: str, args: list[str]):
    """Create a stdio transport for local subprocess communication.

    :param command: The command to execute
    :param args: Arguments to pass to the command
    :return: MCP stdio transport
    """
    server_parameters = StdioServerParameters(
        command=command,
        args=args,
    )
    return stdio_client(server_parameters)


def create_sse_transport(url: str, headers: dict[str, str] | None = None):
    """Create an SSE transport for HTTP Server-Sent Events communication.

    :param url: The SSE endpoint URL (typically ending in /sse)
    :param headers: Optional HTTP headers to include
    :return: MCP SSE transport
    """
    return sse_client(url, headers=headers)


def create_streamable_http_transport(url: str, headers: dict[str, str] | None = None):
    """Create a streamable HTTP transport (newer alternative to SSE).

    :param url: The HTTP endpoint URL (typically ending in /mcp)
    :param headers: Optional HTTP headers to include
    :return: MCP streamable HTTP transport
    """
    return streamablehttp_client(url, headers=headers)


def create_transport(transport_type: str, **kwargs) -> Any:
    """Factory function to create any supported MCP transport type.

    :param transport_type: Type of transport ('stdio', 'sse', 'streamable_http')
    :param kwargs: Transport-specific parameters
    :return: MCP transport instance
    :raises ValueError: If transport_type is not supported
    """
    transport_factories = {
        "stdio": create_stdio_transport,
        "sse": create_sse_transport,
        "streamable_http": create_streamable_http_transport,
    }

    if transport_type not in transport_factories:
        supported_types = ", ".join(transport_factories.keys())
        raise ValueError(
            f"Unsupported transport type '{transport_type}'. Supported types: {supported_types}"
        )

    return transport_factories[transport_type](**kwargs)
