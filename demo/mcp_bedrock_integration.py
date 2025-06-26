#!/usr/bin/env python3
"""
Demo: BedrockAgent with MCP Client Integration

This demonstrates how to use MCP tools with BedrockAgent in different configurations:
1. Single MCP client
2. Multiple MCP clients
3. Mixed MCP and manual tools
"""

import sys
from pathlib import Path

# Add the fence package to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fence.agents.bedrock import BedrockAgent
from fence.models.bedrock.claude import Claude35Sonnet
from fence.tools.mcp.client import MCPClient


def demo_single_mcp_client():
    """Demo: BedrockAgent with a single MCP client."""
    print("=== Demo: Single MCP Client ===")

    client = MCPClient(read_timeout_seconds=30)

    try:
        client.connect("uvx", ["awslabs.aws-documentation-mcp-server@latest"])
        client.list_tools()

        print(f"Connected to MCP server with {len(client.tools)} tools:")
        for tool in client.tools:
            print(f"  - {tool.get_tool_name()}: {tool.get_tool_description()}")

        # Method 1: Create BedrockAgent with single MCP client
        model = Claude35Sonnet(region="us-east-1")
        agent = BedrockAgent(
            identifier="MCP-Agent",
            model=model,
            mcp_clients=client,  # Pass single MCP client - auto-converted to list
            system_message="You are a helpful assistant that can search AWS documentation using MCP tools. Always use the available tools to provide accurate and up-to-date information.",
            log_agentic_response=True,
        )

        print(
            f"\n✅ BedrockAgent created with {len(agent.tools)} tools auto-registered from MCP client"
        )

        # Test the agent with a query
        test_query = "Find information about AWS Lambda functions"
        print(f"\n🤺 Testing agent with query: '{test_query}'")

        response = agent.run(test_query, max_iterations=3)
        print(f"\n📋 Agent Response:\n{response.answer}")

        return response

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return None
    finally:
        client.disconnect()


def demo_multiple_mcp_clients():
    """Demo: BedrockAgent with multiple MCP clients."""
    print("\n=== Demo: Multiple MCP Clients ===")

    # For demonstration, we'll create multiple instances of the same client
    # In practice, you'd connect to different MCP servers
    client1 = MCPClient(read_timeout_seconds=10)
    client2 = MCPClient(read_timeout_seconds=10)

    try:
        # Connect both clients to the same server (for demo purposes)
        client1.connect("uvx", ["awslabs.aws-documentation-mcp-server@latest"])
        client1.list_tools()

        client2.connect("uvx", ["awslabs.aws-documentation-mcp-server@latest"])
        client2.list_tools()

        print(f"Client 1: {len(client1.tools)} tools")
        print(f"Client 2: {len(client2.tools)} tools")

        # Create BedrockAgent with multiple MCP clients
        model = Claude35Sonnet(region="us-east-1")
        agent = BedrockAgent(
            identifier="Multi-MCP-Agent",
            model=model,
            mcp_clients=[client1, client2],  # Multiple clients as list
            system_message="You are a helpful assistant with access to multiple MCP services.",
            log_agentic_response=False,
        )

        print(
            f"✅ Agent created with {len(agent.tools)} tools from {len(agent.mcp_clients)} MCP client(s)"
        )
        return agent

    except Exception as e:
        print(f"❌ Error: {e}")
        return None
    finally:
        client1.disconnect()
        client2.disconnect()


def demo_mixed_setup():
    """Demo: BedrockAgent with both MCP clients and manual tools."""
    print("\n=== Demo: Mixed Setup (MCP + Manual Tools) ===")

    # Import tool here to avoid scoping issues
    from fence.tools.base import tool

    # Create a manual tool
    @tool("A simple calculator")
    def calculator(a: int, b: int, operation: str = "add") -> str:
        """Simple calculator tool.

        :param a: First number
        :param b: Second number
        :param operation: Operation to perform (add or multiply)
        :return: Result of the calculation
        """
        if operation == "add":
            return f"{a} + {b} = {a + b}"
        elif operation == "multiply":
            return f"{a} × {b} = {a * b}"
        else:
            return f"Unknown operation: {operation}"

    client = MCPClient(read_timeout_seconds=10)

    try:
        client.connect("uvx", ["awslabs.aws-documentation-mcp-server@latest"])
        client.list_tools()

        # Create BedrockAgent with both MCP client and manual tools
        model = Claude35Sonnet(region="us-east-1")
        agent = BedrockAgent(
            identifier="Mixed-Agent",
            model=model,
            tools=[calculator],  # Manual tools
            mcp_clients=client,  # MCP client tools
            system_message="You can calculate numbers and search AWS documentation.",
            log_agentic_response=False,
        )

        print(f"✅ Agent created with {len(agent.tools)} total tools:")
        for tool in agent.tools:
            tool_type = "MCP" if hasattr(tool, "mcp_client") else "Manual"
            print(f"  - {tool.get_tool_name()} ({tool_type})")

        return agent

    except Exception as e:
        print(f"❌ Error: {e}")
        return None
    finally:
        client.disconnect()


def demo_manual_tool_registration():
    """Demo: Manual tool registration (alternative approach)."""
    print("\n=== Demo: Manual Tool Registration ===")

    client = MCPClient(read_timeout_seconds=10)

    try:
        client.connect("uvx", ["awslabs.aws-documentation-mcp-server@latest"])
        client.list_tools()

        # Alternative Method: Manual tool registration (still works)
        model = Claude35Sonnet(region="us-east-1")
        agent = BedrockAgent(
            identifier="Manual-Registration-Agent",
            model=model,
            tools=client.tools,  # Manual tool registration
            system_message="You are a helpful assistant with manually registered MCP tools.",
            log_agentic_response=False,
        )

        print(f"✅ Agent created with {len(agent.tools)} manually registered tools")
        print(
            "Note: With manual registration, you manage the MCP client lifecycle separately"
        )

        return agent

    except Exception as e:
        print(f"❌ Error: {e}")
        return None
    finally:
        client.disconnect()


if __name__ == "__main__":
    print("🚀 BedrockAgent + MCP Integration Demos\n")

    # Run different demo scenarios
    single_result = demo_single_mcp_client()
    multiple_result = demo_multiple_mcp_clients()
    mixed_result = demo_mixed_setup()
    manual_result = demo_manual_tool_registration()

    print("\n=== Demo Summary ===")
    demos = [
        ("Single MCP Client", single_result),
        ("Multiple MCP Clients", multiple_result),
        ("Mixed Setup", mixed_result),
        ("Manual Registration", manual_result),
    ]

    for name, result in demos:
        if result:
            tools_count = len(result.tools) if hasattr(result, "tools") else "N/A"
            clients_count = (
                len(result.mcp_clients) if hasattr(result, "mcp_clients") else 0
            )
            print(f"  ✅ {name}: {tools_count} tools, {clients_count} MCP client(s)")
        else:
            print(f"  ❌ {name}: Failed")

    print("\n🎉 All demos completed!")
