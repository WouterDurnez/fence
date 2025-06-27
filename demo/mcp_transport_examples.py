#!/usr/bin/env python3
"""
Demo script showing how to use MCPClient with different transport types.

This script demonstrates connecting to MCP servers using:
- stdio (local subprocess)
- SSE (HTTP Server-Sent Events)
- streamable_http (HTTP streaming)
"""

import logging

from fence.tools.mcp.client import MCPClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_stdio_connection():
    """Example of connecting via stdio to a local MCP server."""
    print("\n=== STDIO Connection Example ===")

    client = MCPClient()

    try:
        # Method 1: Using the general connect method
        client.connect(transport_type="stdio", command="uvx", args=["mcp-server-fetch"])

        # Method 2: Using the convenience method (backward compatibility)
        # client.connect_stdio(command="node", args=["/path/to/mcp-server.js"])

        tools = client.list_tools()
        print(f"Found {len(tools.tools)} tools")

        print(tools)

        # Example tool call
        # result = client.call_tool("example_tool", {"param": "value"})
        # print(f"Tool result: {result}")

    except Exception as e:
        logger.error(f"Error in stdio demo: {e}")
    finally:
        client.disconnect()


def demo_sse_connection():
    """Example of connecting via SSE to a remote MCP server."""
    print("\n=== SSE Connection Example ===")

    client = MCPClient()

    try:
        # Method 1: Using the general connect method
        client.connect(
            transport_type="sse",
            url="https://api.example.com/mcp/sse",
            headers={"Authorization": "Bearer your-token-here"},
        )

        # Method 2: Using the convenience method
        # client.connect_sse(
        #     url="https://api.example.com/mcp/sse",
        #     headers={"Authorization": "Bearer your-token-here"}
        # )

        tools = client.list_tools()
        print(f"Found {len(tools.tools)} tools via SSE")

    except Exception as e:
        logger.error(f"Error in SSE demo: {e}")
    finally:
        client.disconnect()


def demo_streamable_http_connection():
    """Example of connecting via streamable HTTP to a remote MCP server."""
    print("\n=== Streamable HTTP Connection Example ===")

    client = MCPClient()

    try:
        # Method 1: Using the general connect method
        client.connect(
            transport_type="streamable_http",
            url="https://crm-mcp-ai-innovation-day-2025.internal.showpad.io:1337/mcp",
            headers={
                "X-Showpad-context": "eyJpZGVudGl0eSI6eyJ0eXBlIjoidXNlciIsIm9yZ2FuaXNhdGlvbiI6eyJpZCI6IjNhMmVkOGZiNDI4MTFhNzBmMTljZGRkODFiN2FmOTNlNDAwZWMzMjBiZjQ5OWJlYmJmOTBlNGYwODE2YWY4MWIiLCJkYklkIjoiNzcxNDQiLCJ1dWlkIjoiNDhiZGM4NTgtOTZlMC00YWRkLWIxMjQtMThlMzc2YWI0ZWQ5In0sInVzZXIiOnsiaWQiOiI5NDgxYmI1ODBmZTBhMTBjMDJiNWZkOGIzOTE3NjIwNSJ9fSwiYXR0cnMiOnsib2F1dGgyX2NsaWVudF9pZCI6IlNob3dwYWRTZXNzaW9uQ2xpZW50IiwicHJpdmF0ZV9jbGllbnQiOnRydWV9fQ=="
            },
        )

        # Method 2: Using the convenience method
        # client.connect_streamable_http(
        #     url="https://api.example.com/mcp",
        #     headers={"Authorization": "Bearer your-token-here"}
        # )

        tools = client.list_tools()
        print(f"Found {len(tools.tools)} tools via streamable HTTP")
        for tool in tools.tools:
            print(tool.name)

    except Exception as e:
        logger.error(f"Error in streamable HTTP demo: {e}")
    finally:
        client.disconnect()


def demo_error_handling():
    """Example of error handling with unsupported transport types."""
    print("\n=== Error Handling Example ===")

    client = MCPClient()

    try:
        # This will raise a ValueError for unsupported transport type
        client.connect(transport_type="unsupported_transport")
    except ValueError as e:
        print(f"Expected error: {e}")


if __name__ == "__main__":
    print("MCP Client Transport Types Demo")
    print("================================")

    # Note: These examples will fail without actual MCP servers running
    # Uncomment the demos you want to test with real servers

    demo_stdio_connection()
    # demo_sse_connection()
    demo_streamable_http_connection()
    # demo_error_handling()

    print("\nDemo completed!")
