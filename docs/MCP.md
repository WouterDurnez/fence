# 🔌 MCP Integration

Fence has **first-class support for the Model Context Protocol (MCP)**! Connect your agents to MCP servers and automatically expose their tools—no manual configuration needed.

---

## What is MCP?

The [Model Context Protocol](https://modelcontextprotocol.io/) is an open standard for connecting AI assistants to external tools and data sources. Think of it as a universal adapter for AI tools.

**Benefits:**

- 🔌 **Plug-and-play** - Connect to any MCP server
- 🛠️ **Auto-discovery** - Tools are automatically registered
- 🌐 **Standardized** - Works across different AI systems
- 🔒 **Secure** - Built-in authentication and authorization

---

## Quick Start

### Basic MCP Integration

```python
from fence.agents.bedrock import BedrockAgent
from fence.mcp.client import MCPClient
from fence.models.bedrock import Claude37Sonnet

# Create MCP client
mcp_client = MCPClient(
    transport_type="streamable_http",
    url="https://your-mcp-server.com/mcp",
    headers={"Authorization": "Bearer your-token"}
)

# Create agent with MCP tools
agent = BedrockAgent(
    identifier="mcp_agent",
    model=Claude37Sonnet(region="us-east-1"),
    mcp_clients=[mcp_client],  # Tools are auto-registered!
    description="An agent with access to MCP tools"
)

# The agent can now use all tools from the MCP server
result = agent.run("Search for customer data for Acme Corp")
print(result.answer)
```

That's it! The MCP client automatically:

1. Connects to the server
2. Discovers available tools
3. Registers them with the agent
4. Handles tool execution

---

## Transport Types

MCP supports three transport types:

### 1. Streamable HTTP (Recommended)

Modern HTTP streaming transport. Best for production use.

```python
mcp_client = MCPClient(
    transport_type="streamable_http",
    url="https://api.example.com/mcp",
    headers={
        "Authorization": "Bearer your-token",
        "X-Custom-Header": "value"
    },
    read_timeout_seconds=60
)
```

### 2. SSE (Server-Sent Events)

HTTP-based streaming using Server-Sent Events.

```python
mcp_client = MCPClient(
    transport_type="sse",
    url="https://api.example.com/sse",
    headers={"Authorization": "Bearer your-token"},
    read_timeout_seconds=30
)
```

### 3. Stdio (Local Subprocess)

For local MCP servers running as subprocesses.

```python
mcp_client = MCPClient(
    transport_type="stdio",
    command="python",
    args=["path/to/mcp_server.py"]
)
```

---

## Complete Example

Here's a full example with event handlers and error handling:

```python
from fence.agents.bedrock import BedrockAgent
from fence.mcp.client import MCPClient
from fence.models.bedrock import Claude37Sonnet
from fence.utils.logger import setup_logging

# Setup logging
logger = setup_logging(log_level="info")

# Create MCP client
mcp_client = MCPClient(
    transport_type="streamable_http",
    url="https://crm-mcp-server.internal.company.com/mcp",
    headers={
        "X-API-Key": "your-api-key",
        "X-User-Context": "user_session_token"
    },
    read_timeout_seconds=60
)

# Define event handlers
def on_thinking(text: str):
    print(f"🤔 Thinking: {text[:100]}...")

def on_tool_use_start(tool_name: str, parameters: dict):
    print(f"🔧 Using MCP tool: {tool_name}")

def on_tool_use_stop(tool_name: str, parameters: dict, result: dict):
    print(f"✅ Tool completed: {tool_name}")

# Create agent
agent = BedrockAgent(
    identifier="crm_assistant",
    model=Claude37Sonnet(region="us-east-1", cross_region="us"),
    mcp_clients=[mcp_client],
    system_message="""You are a CRM assistant with access to customer data.
    Use the available tools to help users with their queries.
    Always explain what you're doing and why.""",
    event_handlers={
        "on_thinking": on_thinking,
        "on_tool_use_start": on_tool_use_start,
        "on_tool_use_stop": on_tool_use_stop,
    },
    log_agentic_response=True
)

# Use the agent
try:
    result = agent.run("Find all opportunities for Acme Corporation")
    print(f"\n🎉 Answer: {result.answer}")
except Exception as e:
    logger.error(f"Error: {e}")
finally:
    # Clean up
    agent.cleanup()
```

---

## Multiple MCP Servers

Connect to multiple MCP servers simultaneously:

```python
# Create multiple MCP clients
crm_client = MCPClient(
    transport_type="streamable_http",
    url="https://crm-server.com/mcp"
)

analytics_client = MCPClient(
    transport_type="streamable_http",
    url="https://analytics-server.com/mcp"
)

filesystem_client = MCPClient(
    transport_type="stdio",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/path/to/files"]
)

# Agent has access to tools from all servers
agent = BedrockAgent(
    identifier="multi_tool_agent",
    model=Claude37Sonnet(region="us-east-1"),
    mcp_clients=[crm_client, analytics_client, filesystem_client]
)

# Agent can use tools from any server
result = agent.run("""
    1. Search CRM for customer 'Acme Corp'
    2. Get analytics data for their account
    3. Save the report to /reports/acme.txt
""")
```

---

## Context Manager Usage

Use MCP clients as context managers for automatic cleanup:

```python
from fence.mcp.client import MCPClient

# Automatic connection and cleanup
with MCPClient(
    transport_type="streamable_http",
    url="https://api.example.com/mcp"
) as mcp_client:

    agent = BedrockAgent(
        identifier="temp_agent",
        model=Claude37Sonnet(region="us-east-1"),
        mcp_clients=[mcp_client]
    )

    result = agent.run("Do something")
    print(result.answer)

# Connection automatically closed
```

---

## Manual Connection Control

For more control over the connection lifecycle:

```python
mcp_client = MCPClient(
    transport_type="streamable_http",
    url="https://api.example.com/mcp"
)

# Manually connect
mcp_client.connect()

# Check available tools
tools = mcp_client.tools
print(f"Available tools: {[tool.name for tool in tools]}")

# Use with agent
agent = BedrockAgent(
    identifier="agent",
    model=Claude37Sonnet(region="us-east-1"),
    mcp_clients=[mcp_client]
)

result = agent.run("Use the tools")

# Manually disconnect
mcp_client.disconnect()
```

---

## Tool Discovery

MCP tools are automatically discovered and converted to Fence tools:

```python
mcp_client = MCPClient(
    transport_type="streamable_http",
    url="https://api.example.com/mcp"
)

# Connect to discover tools
mcp_client.connect()

# List available tools
for tool in mcp_client.tools:
    print(f"Tool: {tool.name}")
    print(f"  Description: {tool.description}")
    print(f"  Parameters: {tool.parameters}")
```

---

## Error Handling

Handle connection errors gracefully:

```python
from fence.mcp.client import MCPClient

try:
    mcp_client = MCPClient(
        transport_type="streamable_http",
        url="https://api.example.com/mcp",
        read_timeout_seconds=30
    )

    mcp_client.connect()

    agent = BedrockAgent(
        identifier="agent",
        model=Claude37Sonnet(region="us-east-1"),
        mcp_clients=[mcp_client]
    )

    result = agent.run("Query the database")

except ConnectionError as e:
    print(f"Failed to connect to MCP server: {e}")
except TimeoutError as e:
    print(f"MCP server timeout: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
finally:
    if mcp_client:
        mcp_client.disconnect()
```

---

## Configuration Options

### MCPClient Parameters

```python
mcp_client = MCPClient(
    # Transport type
    transport_type="streamable_http",  # or "sse", "stdio"

    # Timeout
    read_timeout_seconds=60,  # Default: 30

    # For HTTP transports (streamable_http, sse)
    url="https://api.example.com/mcp",
    headers={
        "Authorization": "Bearer token",
        "X-Custom-Header": "value"
    },

    # For stdio transport
    command="python",
    args=["server.py", "--port", "8080"]
)
```

---

## Real-World Example: CRM Integration

```python
from fence.agents.bedrock import BedrockAgent
from fence.mcp.client import MCPClient
from fence.models.bedrock import Claude37Sonnet

# Connect to CRM MCP server
mcp_client = MCPClient(
    transport_type="streamable_http",
    url="https://crm.company.com/mcp",
    headers={
        "X-API-Key": "your-api-key",
        "X-User-ID": "user_123"
    }
)

# Create CRM assistant
crm_assistant = BedrockAgent(
    identifier="CRM Assistant",
    model=Claude37Sonnet(region="us-east-1"),
    mcp_clients=[mcp_client],
    system_message="""You are a CRM assistant. You can:
    - Search for accounts, contacts, and opportunities
    - Create and update records
    - Generate reports

    Always confirm before making changes to data."""
)

# Interactive loop
print("CRM Assistant ready! Type 'quit' to exit.")
while True:
    user_input = input("\nYou: ").strip()

    if user_input.lower() in ['quit', 'exit']:
        break

    if not user_input:
        continue

    try:
        result = crm_assistant.run(user_input)
        print(f"\nAssistant: {result.answer}")
    except Exception as e:
        print(f"\nError: {e}")

# Cleanup
crm_assistant.cleanup()
```

---

## Best Practices

### 1. Use Streamable HTTP for Production

```python
# Recommended for production
mcp_client = MCPClient(
    transport_type="streamable_http",
    url="https://api.example.com/mcp"
)
```

### 2. Set Appropriate Timeouts

```python
# Adjust based on your server's response time
mcp_client = MCPClient(
    transport_type="streamable_http",
    url="https://slow-server.com/mcp",
    read_timeout_seconds=120  # 2 minutes for slow operations
)
```

### 3. Always Clean Up

```python
# Use context manager
with MCPClient(...) as client:
    # Use client
    pass

# Or manual cleanup
try:
    client = MCPClient(...)
    # Use client
finally:
    client.disconnect()
```

### 4. Handle Errors Gracefully

```python
try:
    result = agent.run(prompt)
except ConnectionError:
    # Fallback to local tools or retry
    pass
```

---

## Next Steps

- **[Build Multi-Agent Systems →](MULTI_AGENT.md)**
- **[Create Custom Tools →](TOOLS_AND_UTILITIES.md)**
- **[Learn about Agents →](AGENTS.md)**
