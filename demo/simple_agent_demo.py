#!/usr/bin/env python3
"""
Simple Agent Demo using BedrockAgent with MCP Streamable HTTP Client

This demo shows how to create a conversational agent that:
- Connects to a remote MCP server via streamable HTTP
- Uses BedrockAgent with the MCP client to automatically register MCP tools
- Provides a simple interactive chat interface
- Demonstrates how the agent can use MCP tools through natural conversation
"""


from fence.agents.bedrock.agent import BedrockAgent
from fence.mcp.client import MCPClient
from fence.models.bedrock.claude import Claude37Sonnet
from fence.utils.logger import setup_logging

# Configure logging
logger = setup_logging(log_level="kill")


def create_bedrock_agent() -> BedrockAgent:
    """Create a BedrockAgent with the MCP client."""

    # Create the Claude model
    model = Claude37Sonnet(region="us-east-1", cross_region="us")

    # Create the MCP client with connection parameters passed directly to constructor
    mcp_client = MCPClient(
        read_timeout_seconds=60,
        transport_type="streamable_http",
        url="https://crm-mcp-ai-innovation-day-2025.internal.showpad.io:1337/mcp",
        headers={
            "X-Showpad-context": "eyJpZGVudGl0eSI6eyJ0eXBlIjoidXNlciIsIm9yZ2FuaXNhdGlvbiI6eyJpZCI6IjNhMmVkOGZiNDI4MTFhNzBmMTljZGRkODFiN2FmOTNlNDAwZWMzMjBiZjQ5OWJlYmJmOTBlNGYwODE2YWY4MWIiLCJkYklkIjoiNzcxNDQiLCJ1dWlkIjoiNDhiZGM4NTgtOTZlMC00YWRkLWIxMjQtMThlMzc2YWI0ZWQ5In0sInVzZXIiOnsiaWQiOiI5NDgxYmI1ODBmZTBhMTBjMDJiNWZkOGIzOTE3NjIwNSJ9fSwiYXR0cnMiOnsib2F1dGgyX2NsaWVudF9pZCI6IlNob3dwYWRTZXNzaW9uQ2xpZW50IiwicHJpdmF0ZV9jbGllbnQiOnRydWV9fQ=="
        },
    )

    # Define event handlers for better user experience
    def on_thinking(text: str):
        print(f"🤔 Agent thinking: {text[:100]}...")

    def on_tool_use_start(tool_name: str, parameters: dict):
        print(f"🔧 Using tool: {tool_name}")

    def on_tool_use_stop(tool_name: str, parameters: dict, result: dict):
        print(f"✅ Tool {tool_name} completed")

    # Create the agent with the MCP client
    agent = BedrockAgent(
        identifier="Showpad Assist",
        model=model,
        description="A helpful assistant that can access Showpad CRM data and functionality through MCP tools",
        mcp_clients=[mcp_client],
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

    agent = None

    try:
        print("Creating BedrockAgent (will connect to MCP server)...")
        agent = create_bedrock_agent()

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

    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Error during setup: {e}")
        return
    finally:
        # Clean up resources
        if agent:
            try:
                agent.cleanup()
                print("Agent cleaned up successfully")
            except Exception as e:
                logger.warning(f"Error cleaning up agent: {e}")

    print("\nDemo completed!")


if __name__ == "__main__":
    main()
