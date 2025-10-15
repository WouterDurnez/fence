# 🎭 Multi-Agent Systems

Build collaborative agent systems where multiple agents discuss, debate, and solve problems together using the **RoundTable** pattern.

---

## What is RoundTable?

`RoundTable` is a multi-agent orchestration system where multiple `ChatAgent`s participate in a structured conversation. Each agent has its own personality, expertise, and perspective, leading to richer, more nuanced outputs.

**Use cases:**

- 🕵️ Collaborative problem-solving (detective + scientist + witness)
- 💡 Brainstorming sessions (creative + critic + analyst)
- 🎓 Educational discussions (teacher + student + expert)
- 🏛️ Debate simulations (pro + con + moderator)
- 🎬 Story development (writer + editor + character)

---

## Quick Start

### Basic RoundTable

```python
from fence.troupe import RoundTable
from fence.agents import ChatAgent
from fence.models.openai import GPT4omini

# Create specialized agents
detective = ChatAgent(
    identifier="Detective",
    model=GPT4omini(source="roundtable"),
    profile="You are a sharp detective who analyzes clues carefully."
)

scientist = ChatAgent(
    identifier="Scientist",
    model=GPT4omini(source="roundtable"),
    profile="You are a forensic scientist who examines evidence."
)

witness = ChatAgent(
    identifier="Witness",
    model=GPT4omini(source="roundtable"),
    profile="You are a nervous witness who saw something suspicious."
)

# Create the round table
round_table = RoundTable(agents=[detective, scientist, witness])

# Start the discussion
transcript = round_table.run(
    prompt="A valuable painting was stolen from the museum. Let's investigate!",
    max_rounds=3
)

print(transcript)
```

**Output:**

```
Detective: Let me start by examining the crime scene. Witness, what did you see?
Witness: I... I saw someone in a dark coat near the painting around 10 PM.
Scientist: Interesting. I found fingerprints on the frame. Let me analyze them.
Detective: Based on the evidence, I believe we're looking for someone with inside knowledge.
...
```

---

## Creating ChatAgents

`ChatAgent` is designed for conversational interactions:

```python
from fence.agents import ChatAgent
from fence.models.openai import GPT4omini

agent = ChatAgent(
    identifier="Agent Name",           # Display name in conversation
    model=GPT4omini(source="chat"),    # LLM to use
    profile="Agent personality/role",   # System message defining behavior
    memory=FleetingMemory()            # Optional memory backend
)
```

### Agent Profiles

The `profile` parameter defines the agent's personality and expertise:

```python
# Creative agent
creative = ChatAgent(
    identifier="Creative",
    model=GPT4omini(source="roundtable"),
    profile="""You are a creative thinker who loves brainstorming wild ideas.
    You're enthusiastic, optimistic, and always see possibilities."""
)

# Critical agent
critic = ChatAgent(
    identifier="Critic",
    model=GPT4omini(source="roundtable"),
    profile="""You are a critical thinker who spots flaws and risks.
    You're analytical, cautious, and always ask tough questions."""
)

# Practical agent
practical = ChatAgent(
    identifier="Practical",
    model=GPT4omini(source="roundtable"),
    profile="""You are a practical thinker focused on implementation.
    You're pragmatic, detail-oriented, and always consider feasibility."""
)
```

---

## RoundTable Configuration

```python
from fence.troupe import RoundTable
from fence.memory import FleetingMemory

round_table = RoundTable(
    agents=[agent1, agent2, agent3],  # List of ChatAgents
    memory=FleetingMemory()            # Optional shared memory
)

# Run the discussion
transcript = round_table.run(
    prompt="Initial prompt to start the conversation",
    max_rounds=5  # Number of conversation rounds
)
```

### How It Works

1. **Round 1:** Each agent responds to the initial prompt
2. **Round 2:** Each agent responds considering all previous messages
3. **Round N:** Continues until `max_rounds` is reached
4. **Output:** Complete conversation transcript

---

## Examples

### Example 1: Brainstorming Session

```python
from fence.troupe import RoundTable
from fence.agents import ChatAgent
from fence.models.openai import GPT4omini

# Create brainstorming team
ideator = ChatAgent(
    identifier="Ideator",
    model=GPT4omini(source="brainstorm"),
    profile="You generate creative ideas without constraints. Think big!"
)

critic = ChatAgent(
    identifier="Critic",
    model=GPT4omini(source="brainstorm"),
    profile="You evaluate ideas critically, pointing out flaws and risks."
)

implementer = ChatAgent(
    identifier="Implementer",
    model=GPT4omini(source="brainstorm"),
    profile="You focus on how to actually build and implement ideas."
)

# Create round table
brainstorm = RoundTable(agents=[ideator, critic, implementer])

# Run brainstorming session
transcript = brainstorm.run(
    prompt="We need ideas for a new mobile app that helps people learn languages.",
    max_rounds=3
)

print(transcript)
```

### Example 2: Educational Discussion

```python
# Create educational team
teacher = ChatAgent(
    identifier="Teacher",
    model=GPT4omini(source="education"),
    profile="You are a patient teacher who explains concepts clearly."
)

student = ChatAgent(
    identifier="Student",
    model=GPT4omini(source="education"),
    profile="You are a curious student who asks questions and seeks understanding."
)

expert = ChatAgent(
    identifier="Expert",
    model=GPT4omini(source="education"),
    profile="You are a subject matter expert who provides deep insights."
)

# Create classroom
classroom = RoundTable(agents=[teacher, student, expert])

# Run discussion
transcript = classroom.run(
    prompt="Let's discuss how neural networks learn from data.",
    max_rounds=4
)
```

### Example 3: Story Development

```python
# Create writing team
writer = ChatAgent(
    identifier="Writer",
    model=GPT4omini(source="story"),
    profile="You are a creative writer who develops engaging narratives."
)

editor = ChatAgent(
    identifier="Editor",
    model=GPT4omini(source="story"),
    profile="You are an editor who improves structure and clarity."
)

character_expert = ChatAgent(
    identifier="Character Expert",
    model=GPT4omini(source="story"),
    profile="You specialize in character development and dialogue."
)

# Create writers' room
writers_room = RoundTable(agents=[writer, editor, character_expert])

# Develop story
transcript = writers_room.run(
    prompt="Let's develop a story about a time traveler who accidentally changes history.",
    max_rounds=3
)
```

### Example 4: Debate Simulation

```python
# Create debate team
pro = ChatAgent(
    identifier="Pro",
    model=GPT4omini(source="debate"),
    profile="You argue in favor of the proposition with strong evidence."
)

con = ChatAgent(
    identifier="Con",
    model=GPT4omini(source="debate"),
    profile="You argue against the proposition with counterarguments."
)

moderator = ChatAgent(
    identifier="Moderator",
    model=GPT4omini(source="debate"),
    profile="You moderate the debate, summarize points, and ask clarifying questions."
)

# Create debate
debate = RoundTable(agents=[pro, con, moderator])

# Run debate
transcript = debate.run(
    prompt="Should artificial intelligence be regulated by governments?",
    max_rounds=4
)
```

---

## Advanced Patterns

### Persistent Memory

Use persistent memory to maintain context across sessions:

```python
from fence.memory import DynamoDBMemory

# Create shared memory
shared_memory = DynamoDBMemory(
    table_name="roundtable_memory",
    primary_key_name="session_id",
    primary_key_value="brainstorm_session_123"
)

# Create round table with shared memory
round_table = RoundTable(
    agents=[agent1, agent2, agent3],
    memory=shared_memory
)

# First session
round_table.run(prompt="Let's discuss project ideas", max_rounds=2)

# Later session - agents remember previous discussion
round_table.run(prompt="Let's continue our discussion", max_rounds=2)
```

### Dynamic Agent Addition

```python
# Start with initial agents
round_table = RoundTable(agents=[agent1, agent2])

# Add more agents later
round_table.add_agent(agent3)
round_table.add_agent(agent4)

# Run with all agents
transcript = round_table.run(prompt="Discuss the topic", max_rounds=3)
```

### Accessing Conversation History

```python
round_table = RoundTable(agents=[agent1, agent2, agent3])

# Run discussion
transcript = round_table.run(prompt="Discuss AI ethics", max_rounds=3)

# Access individual turns
for turn in round_table.conversation_history:
    print(f"{turn.agent_name}: {turn.message}")
```

---

## Best Practices

### 1. Create Diverse Agents

```python
# Good - diverse perspectives
agents = [
    ChatAgent(identifier="Optimist", profile="You see opportunities..."),
    ChatAgent(identifier="Pessimist", profile="You see risks..."),
    ChatAgent(identifier="Realist", profile="You see facts...")
]

# Bad - too similar
agents = [
    ChatAgent(identifier="Agent1", profile="You are helpful"),
    ChatAgent(identifier="Agent2", profile="You are helpful"),
    ChatAgent(identifier="Agent3", profile="You are helpful")
]
```

### 2. Set Appropriate Round Limits

```python
# Quick discussion
quick_chat = round_table.run(prompt="...", max_rounds=2)

# Deep discussion
deep_dive = round_table.run(prompt="...", max_rounds=5)

# Be careful with too many rounds (can get repetitive)
# Avoid: max_rounds=20
```

### 3. Use Clear Agent Identifiers

```python
# Good - clear roles
ChatAgent(identifier="Lead Developer", ...)
ChatAgent(identifier="UX Designer", ...)
ChatAgent(identifier="Product Manager", ...)

# Bad - unclear
ChatAgent(identifier="Agent1", ...)
ChatAgent(identifier="Bob", ...)
```

### 4. Craft Specific Profiles

```python
# Good - specific and actionable
profile = """You are a cybersecurity expert with 15 years of experience.
You focus on identifying vulnerabilities and suggesting mitigations.
You always consider both technical and business impacts."""

# Bad - too vague
profile = "You are an expert who knows things."
```

---

## Combining with Other Features

### RoundTable + Tools

Give agents access to tools:

```python
from fence.agents import Agent  # Note: Agent, not ChatAgent
from fence.tools.math import CalculatorTool

# Create agent with tools (use Agent, not ChatAgent)
analyst = Agent(
    identifier="Analyst",
    model=GPT4omini(source="roundtable"),
    tools=[CalculatorTool()],
    description="You analyze data and perform calculations"
)

# Note: RoundTable is designed for ChatAgents
# For tool-using agents, consider hierarchical delegation instead
```

### Hierarchical Multi-Agent Systems

Combine RoundTable with agent delegation:

```python
# Create a RoundTable for discussion
discussion_agents = [
    ChatAgent(identifier="Researcher", ...),
    ChatAgent(identifier="Analyst", ...)
]
discussion = RoundTable(agents=discussion_agents)

# Create a coordinator agent that can use the discussion
coordinator = Agent(
    identifier="Coordinator",
    model=GPT4omini(source="app"),
    delegates=[],  # Could delegate to other agents
    description="Coordinates team discussions"
)

# Coordinator runs the discussion
discussion_result = discussion.run(
    prompt="Analyze market trends",
    max_rounds=3
)

# Coordinator uses the discussion output
final_result = coordinator.run(
    f"Based on this discussion: {discussion_result}, create an action plan"
)
```

---

## Real-World Example: Product Review

```python
from fence.troupe import RoundTable
from fence.agents import ChatAgent
from fence.models.bedrock import ClaudeHaiku

# Create review team
technical_reviewer = ChatAgent(
    identifier="Technical Reviewer",
    model=ClaudeHaiku(),
    profile="""You are a technical expert who evaluates products based on:
    - Technical specifications
    - Performance metrics
    - Build quality
    You provide detailed technical analysis."""
)

user_experience_reviewer = ChatAgent(
    identifier="UX Reviewer",
    model=ClaudeHaiku(),
    profile="""You are a UX expert who evaluates products based on:
    - Ease of use
    - User interface design
    - Accessibility
    You focus on the user's perspective."""
)

value_reviewer = ChatAgent(
    identifier="Value Reviewer",
    model=ClaudeHaiku(),
    profile="""You are a value analyst who evaluates products based on:
    - Price vs. features
    - Long-term value
    - Alternatives comparison
    You help users make informed purchasing decisions."""
)

# Create review panel
review_panel = RoundTable(agents=[
    technical_reviewer,
    user_experience_reviewer,
    value_reviewer
])

# Review a product
product_review = review_panel.run(
    prompt="""Let's review the new iPhone 15 Pro.
    Consider its technical specs, user experience, and value for money.""",
    max_rounds=3
)

print(product_review)
```

---

## Next Steps

- **[Learn about Agents →](AGENTS.md)**
- **[Explore MCP Integration →](MCP.md)**
- **[Configure Memory →](MEMORY.md)**
