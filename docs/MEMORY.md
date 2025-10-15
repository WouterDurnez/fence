# 🧠 Memory Systems

Agents need memory to maintain context across conversations. Fence provides multiple memory backends for different use cases.

---

## Memory Types

Fence supports four memory backends:

1. **`FleetingMemory`** - In-memory (ephemeral, default)
2. **`DynamoDBMemory`** - AWS DynamoDB (persistent, scalable)
3. **`SQLiteMemory`** - Local SQLite database (persistent, local)
4. **`AgentCoreMemory`** - AWS Agent Core integration

---

## FleetingMemory

In-memory storage that's lost when the process ends. Perfect for development and short-lived conversations.

### Basic Usage

```python
from fence.agents import Agent
from fence.memory import FleetingMemory
from fence.models.openai import GPT4omini

# Create memory
memory = FleetingMemory()

# Create agent with memory
agent = Agent(
    identifier="chatbot",
    model=GPT4omini(source="app"),
    memory=memory
)

# Multi-turn conversation
agent.run("My name is Alice")
agent.run("I like Python programming")
response = agent.run("What's my name and what do I like?")
# "Your name is Alice and you like Python programming"
```

### When to Use

- ✅ Development and testing
- ✅ Short-lived conversations
- ✅ Stateless applications
- ❌ Production with persistent sessions
- ❌ Multi-instance deployments

---

## DynamoDBMemory

Persistent storage in AWS DynamoDB. Perfect for production applications with distributed agents.

### Setup

First, create a DynamoDB table:

```bash
aws dynamodb create-table \
    --table-name agent-memory \
    --attribute-definitions \
        AttributeName=session_id,AttributeType=S \
    --key-schema \
        AttributeName=session_id,KeyType=HASH \
    --billing-mode PAY_PER_REQUEST
```

### Basic Usage

```python
from fence.agents import Agent
from fence.memory import DynamoDBMemory
from fence.models.openai import GPT4omini

# Create DynamoDB memory
memory = DynamoDBMemory(
    table_name="agent-memory",
    primary_key_name="session_id",
    primary_key_value="user_123_session_456"
)

# Create agent
agent = Agent(
    identifier="persistent_agent",
    model=GPT4omini(source="app"),
    memory=memory
)

# Conversations persist across runs
agent.run("Remember: my favorite color is blue")

# Later, in a different process...
agent = Agent(
    identifier="persistent_agent",
    model=GPT4omini(source="app"),
    memory=DynamoDBMemory(
        table_name="agent-memory",
        primary_key_name="session_id",
        primary_key_value="user_123_session_456"  # Same session
    )
)

response = agent.run("What's my favorite color?")
# "Your favorite color is blue"
```

### Configuration

```python
memory = DynamoDBMemory(
    table_name="agent-memory",           # DynamoDB table name
    primary_key_name="session_id",       # Primary key attribute name
    primary_key_value="unique_session",  # Unique session identifier
    region="us-east-1"                   # AWS region (optional)
)
```

### Session Management

```python
# Different users/sessions
user1_memory = DynamoDBMemory(
    table_name="agent-memory",
    primary_key_name="session_id",
    primary_key_value="user_1_session"
)

user2_memory = DynamoDBMemory(
    table_name="agent-memory",
    primary_key_name="session_id",
    primary_key_value="user_2_session"
)

# Each user has isolated memory
agent1 = Agent(identifier="agent", model=GPT4omini(source="app"), memory=user1_memory)
agent2 = Agent(identifier="agent", model=GPT4omini(source="app"), memory=user2_memory)
```

### When to Use

- ✅ Production applications
- ✅ Multi-instance deployments
- ✅ Long-term conversation history
- ✅ Scalable storage needs
- ❌ Local development (use FleetingMemory)
- ❌ Offline applications

---

## SQLiteMemory

Local persistent storage using SQLite. Perfect for single-instance applications and local development.

### Basic Usage

```python
from fence.agents import Agent
from fence.memory import SQLiteMemory
from fence.models.openai import GPT4omini

# Create SQLite memory
memory = SQLiteMemory(
    db_path="agent_memory.db",
    session_id="user_session_123"
)

# Create agent
agent = Agent(
    identifier="local_agent",
    model=GPT4omini(source="app"),
    memory=memory
)

# Conversations persist locally
agent.run("My favorite book is 1984")

# Later...
agent = Agent(
    identifier="local_agent",
    model=GPT4omini(source="app"),
    memory=SQLiteMemory(
        db_path="agent_memory.db",
        session_id="user_session_123"  # Same session
    )
)

response = agent.run("What's my favorite book?")
# "Your favorite book is 1984"
```

### Configuration

```python
memory = SQLiteMemory(
    db_path="path/to/database.db",  # Database file path
    session_id="unique_session"      # Unique session identifier
)
```

### Multiple Sessions

```python
# Different sessions in the same database
session1 = SQLiteMemory(db_path="memory.db", session_id="session_1")
session2 = SQLiteMemory(db_path="memory.db", session_id="session_2")

agent1 = Agent(identifier="agent", model=GPT4omini(source="app"), memory=session1)
agent2 = Agent(identifier="agent", model=GPT4omini(source="app"), memory=session2)
```

### When to Use

- ✅ Local development
- ✅ Single-instance applications
- ✅ Desktop applications
- ✅ Offline applications
- ❌ Multi-instance deployments
- ❌ High-scale production

---

## AgentCoreMemory

Integration with AWS Agent Core for enterprise-grade memory management.

### Basic Usage

```python
from fence.agents import Agent
from fence.memory import AgentCoreMemory
from fence.models.openai import GPT4omini

# Create AgentCore memory
memory = AgentCoreMemory(
    agent_id="agent_123",
    session_id="session_456"
)

# Create agent
agent = Agent(
    identifier="enterprise_agent",
    model=GPT4omini(source="app"),
    memory=memory
)
```

### When to Use

- ✅ AWS Agent Core integration
- ✅ Enterprise deployments
- ✅ Advanced memory features
- ❌ Simple applications

---

## Memory Operations

All memory backends support the same interface:

### Adding Messages

```python
# Add user message
memory.add_message(role="user", content="Hello!")

# Add assistant message
memory.add_message(role="assistant", content="Hi there!")

# Add system message
memory.add_message(role="system", content="You are a helpful assistant")
```

### Setting System Message

```python
# Set or update system message
memory.set_system_message("You are a helpful coding assistant")
```

### Getting Messages

```python
# Get all messages
messages = memory.get_messages()

for message in messages:
    print(f"{message.role}: {message.content}")
```

### Clearing Memory

```python
# Clear all messages (keeps system message)
memory.clear()
```

---

## Advanced Patterns

### Memory with Max Size

Limit memory size to prevent context overflow:

```python
from fence.agents import Agent

agent = Agent(
    identifier="limited_memory_agent",
    model=GPT4omini(source="app"),
    memory=FleetingMemory(),
    max_memory_size=1000  # Max tokens in memory
)
```

### Shared Memory Across Agents

```python
# Create shared memory
shared_memory = DynamoDBMemory(
    table_name="shared-memory",
    primary_key_name="session_id",
    primary_key_value="team_session"
)

# Multiple agents share the same memory
agent1 = Agent(identifier="agent1", model=GPT4omini(source="app"), memory=shared_memory)
agent2 = Agent(identifier="agent2", model=GPT4omini(source="app"), memory=shared_memory)

# Both agents see the same conversation history
agent1.run("The password is 'secret123'")
agent2.run("What's the password?")  # "The password is 'secret123'"
```

### Memory Prefill

Start conversations with predefined context:

```python
agent = Agent(
    identifier="prefilled_agent",
    model=GPT4omini(source="app"),
    memory=FleetingMemory(),
    prefill="I understand. I'll help you with that."
)

# Agent starts with the prefilled message
```

---

## Best Practices

### 1. Choose the Right Backend

```python
# Development
dev_memory = FleetingMemory()

# Production (AWS)
prod_memory = DynamoDBMemory(
    table_name="prod-agent-memory",
    primary_key_name="session_id",
    primary_key_value=f"user_{user_id}"
)

# Local production
local_prod_memory = SQLiteMemory(
    db_path="/var/app/memory.db",
    session_id=f"user_{user_id}"
)
```

### 2. Use Unique Session IDs

```python
import uuid

# Good - unique per user/session
session_id = f"user_{user_id}_session_{uuid.uuid4()}"

memory = DynamoDBMemory(
    table_name="memory",
    primary_key_name="session_id",
    primary_key_value=session_id
)

# Bad - shared across users
memory = DynamoDBMemory(
    table_name="memory",
    primary_key_name="session_id",
    primary_key_value="shared_session"  # Everyone sees same history!
)
```

### 3. Handle Memory Cleanup

```python
# Clear old sessions periodically
def cleanup_old_sessions(memory: DynamoDBMemory, max_age_days: int):
    # Implementation depends on your backend
    pass

# Or use TTL in DynamoDB
# Set TTL attribute when creating the table
```

### 4. Monitor Memory Size

```python
# Check memory size
messages = memory.get_messages()
total_tokens = sum(len(msg.content.split()) for msg in messages)

if total_tokens > 10000:
    # Summarize or truncate old messages
    memory.clear()
    memory.add_message(
        role="system",
        content="Previous conversation summarized: ..."
    )
```

---

## Real-World Example: Customer Support Bot

```python
from fence.agents import Agent
from fence.memory import DynamoDBMemory
from fence.models.bedrock import Claude35Sonnet

def create_support_agent(user_id: str, session_id: str) -> Agent:
    """Create a customer support agent with persistent memory."""

    # Create persistent memory for this user session
    memory = DynamoDBMemory(
        table_name="customer-support-memory",
        primary_key_name="session_id",
        primary_key_value=f"user_{user_id}_session_{session_id}"
    )

    # Create agent
    agent = Agent(
        identifier="support_agent",
        model=Claude35Sonnet(region="us-east-1"),
        memory=memory,
        role="""You are a helpful customer support agent.
        You remember previous conversations with the customer.
        Always be polite and professional.""",
        max_memory_size=5000  # Limit context size
    )

    return agent

# Usage
agent = create_support_agent(user_id="12345", session_id="abc-def")

# First interaction
agent.run("I'm having trouble with my order #98765")

# Later interaction (same session)
agent.run("What was my order number again?")
# "Your order number is #98765"

# Different session
new_agent = create_support_agent(user_id="12345", session_id="xyz-123")
new_agent.run("Hello")  # Fresh conversation, no memory of previous session
```

---

## Next Steps

- **[Learn about Agents →](AGENTS.md)**
- **[Build Multi-Agent Systems →](MULTI_AGENT.md)**
- **[Explore Tools →](TOOLS_AND_UTILITIES.md)**
