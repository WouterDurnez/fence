#!/usr/bin/env python3
"""
Simple Agent Demo using BedrockAgent with MCP Streamable HTTP Client

This demo shows how to create a conversational agent that:
- Connects to a remote MCP server via streamable HTTP
- Uses BedrockAgent with the MCP client to automatically register MCP tools
- Provides a simple interactive chat interface
- Demonstrates how the agent can use MCP tools through natural conversation
"""

import logging

from fence.agents.bedrock.agent import BedrockAgent
from fence.models.bedrock.claude import Claude37Sonnet
from fence.tools.mcp.client import MCPClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_mcp_client() -> MCPClient:
    """Create and connect to the MCP server via streamable HTTP."""
    client = MCPClient()

    try:
        client.connect(
            transport_type="streamable_http",
            url="https://crm-mcp-ai-innovation-day-2025.internal.showpad.io:1337/mcp",
            headers={
                "X-Showpad-context": "eyJpZGVudGl0eSI6eyJ0eXBlIjoidXNlciIsIm9yZ2FuaXNhdGlvbiI6eyJpZCI6IjNhMmVkOGZiNDI4MTFhNzBmMTljZGRkODFiN2FmOTNlNDAwZWMzMjBiZjQ5OWJlYmJmOTBlNGYwODE2YWY4MWIiLCJkYklkIjoiNzcxNDQiLCJ1dWlkIjoiNDhiZGM4NTgtOTZlMC00YWRkLWIxMjQtMThlMzc2YWI0ZWQ5In0sInVzZXIiOnsiaWQiOiI5NDgxYmI1ODBmZTBhMTBjMDJiNWZkOGIzOTE3NjIwNSJ9fSwiYXR0cnMiOnsib2F1dGgyX2NsaWVudF9pZCI6IlNob3dwYWRTZXNzaW9uQ2xpZW50IiwicHJpdmF0ZV9jbGllbnQiOnRydWV9fQ=="
            },
        )

        tools = client.list_tools()
        logger.info(
            f"Successfully connected to MCP server with {len(tools.tools)} tools"
        )

        # Log available tools
        for tool in tools.tools:
            logger.info(f"  - {tool.name}: {tool.description}")

        return client

    except Exception as e:
        logger.error(f"Failed to connect to MCP server: {e}")
        raise


def create_bedrock_agent(mcp_client: MCPClient) -> BedrockAgent:
    """Create a BedrockAgent with the MCP client."""

    # Create the Claude model
    model = Claude37Sonnet(region="us-east-1", cross_region="us")

    # Define event handlers for better user experience
    def on_thinking(text: str):
        print(f"🤔 Agent thinking: {text[:100]}...")

    def on_tool_use_start(tool_name: str, parameters: dict):
        print(f"🔧 Using tool: {tool_name}")

    def on_tool_use_stop(tool_name: str, parameters: dict, result: dict):
        print(f"✅ Tool {tool_name} completed")

    # Create the agent with the MCP client
    agent = BedrockAgent(
        identifier="ShowpadAgent",
        model=model,
        description="A helpful assistant that can access Showpad CRM data and functionality through MCP tools",
        mcp_clients=[mcp_client],  # Pass the MCP client here
        system_message="""You are a helpful assistant with access to Showpad CRM tools.
        You can help users with CRM-related tasks like looking up accounts, contacts, opportunities, and more.
        Always use the available tools to provide accurate and up-to-date information.
        When using tools, explain what you're doing and why.""",
        event_handlers={
            "on_thinking": on_thinking,
            "on_tool_use_start": on_tool_use_start,
            "on_tool_use_stop": on_tool_use_stop,
        },
        log_agentic_response=True,
        are_you_serious=False,
    )

    return agent


def main():
    """Main demo function."""
    print("🚀 Simple Agent Demo with BedrockAgent and MCP")
    print("=" * 50)

    # Create MCP client
    print("Connecting to MCP server...")
    mcp_client = create_mcp_client()

    # Create Bedrock agent
    print("Creating BedrockAgent...")
    agent = create_bedrock_agent(mcp_client)

    print("\n🎯 Agent ready! You can now chat with the agent.")
    print("The agent has access to Showpad CRM tools through MCP.")
    print("Type 'quit' or 'exit' to stop the demo.\n")

    try:
        while True:
            # Get user input
            user_input = input("You: ").strip()

            if user_input.lower() in ["quit", "exit", "q"]:
                break

            if not user_input:
                continue

            print(f"\n🤖 Agent processing: {user_input}")
            print("-" * 30)

            try:
                # Run the agent
                response = agent.run(user_input, max_iterations=5)

                print(f"\n🎉 Agent: {response.answer}")
                print("-" * 50)

            except Exception as e:
                print(f"❌ Error: {e}")
                logger.error(f"Error during agent execution: {e}")

    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user")

    finally:
        # Clean up
        print("\nCleaning up...")
        mcp_client.disconnect()
        print("✅ MCP client disconnected")
        print("Demo completed!")


if __name__ == "__main__":
    main()
