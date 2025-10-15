# 🤖 Agents

Agents are where Fence really shines! We've built **production-ready agentic systems** using the ReAct (Reasoning + Acting) pattern. Agents can think, use tools, delegate to other agents, and maintain memory across conversations.

---

## Agent Types

Fence provides three types of agents:

1. **`Agent`** - Classic ReAct agent with tool use and multi-level delegation
2. **`BedrockAgent`** - Native Bedrock tool calling with streaming support
3. **`ChatAgent`** - Conversational agent for multi-agent systems

---

## ReAct Agent

The classic agent implementation using the ReAct pattern: **Thought → Action → Observation → Answer**.

### Basic Example

```python
from fence.agents import Agent
from fence.models.openai import GPT4omini
from fence.tools.math import CalculatorTool

# Create an agent with tools
agent = Agent(
    identifier="math_wizard",
    model=GPT4omini(source="demo"),
    description="An agent that can perform calculations",
    tools=[CalculatorTool()],
    max_iterations=5
)

# Ask it something that requires tool use
result = agent.run("What is 1337 * 42 + 999?")
print(result)
```

**What happens:**

1. **[THOUGHT]** - Agent reasons about the problem
2. **[ACTION]** - Agent uses the calculator tool
3. **[OBSERVATION]** - Agent sees the tool result
4. **[ANSWER]** - Agent provides the final answer

### Agent Configuration

```python
from fence.agents import Agent
from fence.memory import DynamoDBMemory

agent = Agent(
    identifier="my_agent",              # Unique identifier
    model=GPT4omini(source="app"),      # LLM to use
    description="A helpful assistant",   # Agent description
    role="You are an expert analyst",    # System message role
    tools=[CalculatorTool()],            # Available tools
    delegates=[],                        # Other agents to delegate to
    memory=DynamoDBMemory(...),          # Memory backend
    environment={"api_key": "..."},      # Environment variables
    prefill="Let me help you with that", # Assistant prefill
    max_iterations=5,                    # Max reasoning loops
    iteration_timeout=30.0,              # Timeout per iteration
    log_agentic_response=True,           # Log agent thoughts
    are_you_serious=False                # Fun logging mode
)
```

### Multi-Level Agent Delegation

Agents can delegate to other specialized agents!

```python
from fence.agents import Agent
from fence.tools.math import CalculatorTool, PrimeTool
from fence.tools.text import SecretStringTool

# Create a specialist agent
math_agent = Agent(
    identifier="mathematician",
    model=GPT4omini(source="agent"),
    description="An agent specialized in mathematical operations",
    tools=[CalculatorTool(), PrimeTool()]
)

# Create a coordinator agent
coordinator = Agent(
    identifier="coordinator",
    model=GPT4omini(source="agent"),
    description="A coordinator that can delegate to specialists",
    delegates=[math_agent],
    tools=[SecretStringTool()]
)

# The coordinator will delegate math questions to the specialist
result = coordinator.run("Is 17 a prime number? Also, what's the secret string?")
```

**What happens:**

1. Coordinator receives the question
2. Recognizes it needs math expertise
3. **[DELEGATE]** to the math_agent
4. Math agent uses PrimeTool
5. Coordinator uses SecretStringTool
6. Coordinator combines results and answers

### Complex Hierarchies

Build multi-level agent hierarchies:

```python
# Level 3: Specialist agents
calculator_agent = Agent(
    identifier="calculator",
    model=GPT4omini(source="agent"),
    tools=[CalculatorTool()],
    description="Performs calculations"
)

text_agent = Agent(
    identifier="text_processor",
    model=GPT4omini(source="agent"),
    tools=[TextInverterTool()],
    description="Processes text"
)

# Level 2: Department coordinators
math_coordinator = Agent(
    identifier="math_dept",
    model=GPT4omini(source="agent"),
    delegates=[calculator_agent],
    description="Coordinates mathematical tasks"
)

# Level 1: Master coordinator
master = Agent(
    identifier="master",
    model=GPT4omini(source="agent"),
    delegates=[math_coordinator, text_agent],
    description="Master coordinator for all tasks"
)

# The master will delegate down the hierarchy as needed
result = master.run("Calculate 42 * 1337, then invert the result as text")
```

---

## BedrockAgent

Native Bedrock agent with streaming support and event handlers. Uses Bedrock's native tool calling API for better performance.

### Basic Example

```python
from fence.agents.bedrock import BedrockAgent
from fence.models.bedrock import Claude35Sonnet
from fence.tools.text import TextInverterTool

# Create Bedrock agent
agent = BedrockAgent(
    identifier="text_processor",
    model=Claude35Sonnet(region="us-east-1"),
    tools=[TextInverterTool()],
    description="An agent that processes text"
)

result = agent.run("Invert the text: Hello World!")
print(result.answer)
```

### Event Handlers

BedrockAgent supports event handlers for real-time visibility:

```python
from fence.agents.bedrock import BedrockAgent

# Define event handlers
def on_thinking(text: str):
    print(f"🤔 Agent thinking: {text[:100]}...")

def on_tool_use_start(tool_name: str, parameters: dict):
    print(f"🔧 Using tool: {tool_name}")
    print(f"   Parameters: {parameters}")

def on_tool_use_stop(tool_name: str, parameters: dict, result: dict):
    print(f"✅ Tool {tool_name} completed")
    print(f"   Result: {result}")

def on_delegate_start(delegate_name: str, prompt: str):
    print(f"👥 Delegating to: {delegate_name}")

def on_delegate_stop(delegate_name: str, result: str):
    print(f"✅ Delegate {delegate_name} completed")

# Create agent with handlers
agent = BedrockAgent(
    identifier="monitored_agent",
    model=Claude35Sonnet(region="us-east-1"),
    tools=[CalculatorTool()],
    event_handlers={
        "on_thinking": on_thinking,
        "on_tool_use_start": on_tool_use_start,
        "on_tool_use_stop": on_tool_use_stop,
        "on_delegate_start": on_delegate_start,
        "on_delegate_stop": on_delegate_stop,
    }
)

result = agent.run("What is 42 * 1337?")
```

### BedrockAgent with Delegation

```python
from fence.agents.bedrock import BedrockAgent

# Create specialist
specialist = BedrockAgent(
    identifier="math_specialist",
    model=Claude35Sonnet(region="us-east-1"),
    tools=[CalculatorTool(), PrimeTool()],
    description="Specialized in mathematical operations"
)

# Create coordinator
coordinator = BedrockAgent(
    identifier="coordinator",
    model=Claude35Sonnet(region="us-east-1"),
    delegates=[specialist],
    description="Coordinates tasks and delegates when needed"
)

result = coordinator.run("Is 97 a prime number?")
```

### Custom System Messages

```python
agent = BedrockAgent(
    identifier="custom_agent",
    model=Claude35Sonnet(region="us-east-1"),
    system_message="""You are a helpful assistant specialized in data analysis.
    You have access to various tools and should use them when appropriate.
    Always explain your reasoning before taking action.""",
    tools=[CalculatorTool()]
)
```

---

## ChatAgent

Conversational agent designed for multi-agent systems and chat interfaces.

### Basic Example

```python
from fence.agents import ChatAgent
from fence.models.openai import GPT4omini

# Create a chat agent
agent = ChatAgent(
    identifier="Assistant",
    model=GPT4omini(source="chat"),
    profile="You are a friendly and helpful assistant with a sense of humor."
)

# Have a conversation
response = agent.run("Tell me a joke about programming")
print(response)
```

### Chat with Memory

```python
from fence.memory import FleetingMemory

memory = FleetingMemory()

agent = ChatAgent(
    identifier="Chatbot",
    model=GPT4omini(source="chat"),
    profile="You are a knowledgeable assistant",
    memory=memory
)

# Multi-turn conversation
agent.run("My name is Alice")
agent.run("I like Python programming")
response = agent.run("What's my name and what do I like?")
# Agent remembers: "Your name is Alice and you like Python programming"
```

---

## Agent Memory

Agents support multiple memory backends for persistent conversations.

### In-Memory (Default)

```python
from fence.memory import FleetingMemory

agent = Agent(
    identifier="ephemeral_agent",
    model=GPT4omini(source="app"),
    memory=FleetingMemory()  # Default, lost when process ends
)
```

### DynamoDB (Persistent)

```python
from fence.memory import DynamoDBMemory

memory = DynamoDBMemory(
    table_name="agent_conversations",
    primary_key_name="session_id",
    primary_key_value="user_123_session_456"
)

agent = Agent(
    identifier="persistent_agent",
    model=GPT4omini(source="app"),
    memory=memory
)

# Conversations persist across runs
agent.run("Remember: my favorite color is blue")
# Later, in a new process...
agent.run("What's my favorite color?")  # "Your favorite color is blue"
```

### SQLite (Local Persistent)

```python
from fence.memory import SQLiteMemory

memory = SQLiteMemory(
    db_path="agent_memory.db",
    session_id="user_session_123"
)

agent = Agent(
    identifier="local_agent",
    model=GPT4omini(source="app"),
    memory=memory
)
```

---

## Environment Variables

Pass environment variables to agents and their tools/delegates:

```python
agent = Agent(
    identifier="api_agent",
    model=GPT4omini(source="app"),
    environment={
        "api_key": "secret_key_123",
        "base_url": "https://api.example.com",
        "user_id": "user_456"
    },
    tools=[CustomAPITool()]  # Tool can access environment
)

# Tools receive environment in their run() method
class CustomAPITool(BaseTool):
    def run(self, query: str, **kwargs) -> str:
        env = kwargs.get('environment', {})
        api_key = env.get('api_key')
        # Use the API key...
```

---

## Best Practices

### 1. Choose the Right Agent Type

- **Use `Agent`** for complex multi-agent hierarchies with OpenAI/Anthropic
- **Use `BedrockAgent`** for production AWS workloads with streaming
- **Use `ChatAgent`** for conversational interfaces and multi-agent discussions

### 2. Set Appropriate Limits

```python
agent = Agent(
    identifier="safe_agent",
    model=GPT4omini(source="app"),
    max_iterations=5,        # Prevent infinite loops
    iteration_timeout=30.0   # Timeout per iteration
)
```

### 3. Use Descriptive Identifiers and Descriptions

```python
# Good
agent = Agent(
    identifier="customer_support_agent",
    description="Handles customer inquiries about orders, returns, and shipping"
)

# Bad
agent = Agent(
    identifier="agent1",
    description="Does stuff"
)
```

### 4. Monitor with Event Handlers (BedrockAgent)

```python
def log_tool_use(tool_name: str, parameters: dict):
    logger.info(f"Tool used: {tool_name}", extra={"params": parameters})

agent = BedrockAgent(
    identifier="monitored",
    model=Claude35Sonnet(region="us-east-1"),
    event_handlers={"on_tool_use_start": log_tool_use}
)
```

---

## Next Steps

- **[Learn about MCP Integration →](MCP.md)**
- **[Build Multi-Agent Systems →](MULTI_AGENT.md)**
- **[Create Custom Tools →](TOOLS_AND_UTILITIES.md)**
- **[Configure Memory →](MEMORY.md)**
