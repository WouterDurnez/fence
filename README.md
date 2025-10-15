<img src="https://github.com/WouterDurnez/fence/blob/main/docs/logo.png?raw=true" alt="tests" height="200"/>

[![Python](https://img.shields.io/pypi/pyversions/fence-llm)](https://pypi.org/project/fence-llm/)
[![Test Status](https://github.com/WouterDurnez/fence/actions/workflows/ci-pipeline.yaml/badge.svg)](https://github.com/WouterDurnez/fence/actions)
[![codecov](https://codecov.io/gh/WouterDurnez/fence/branch/main/graph/badge.svg?token=QZQZQZQZQZ)](https://codecov.io/gh/WouterDurnez/fence)
[![PyPI version](https://badge.fury.io/py/fence-llm.svg)](https://badge.fury.io/py/fence-llm)
[![Documentation Status](https://readthedocs.org/projects/fence-llm/badge/?version=latest)](https://fence-llm.readthedocs.io/en/latest/?badge=latest)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](code_of_conduct.md)

# 🤺 Fence

**The Bloat Moat!** A lightweight, production-ready library for LLM communication and agentic workflows. Born from the need for something simpler than LangChain, Fence gives you powerful LLM orchestration without the heavyweight dependencies.

Think of it as the Swiss Army knife for LLM interactions—sharp, reliable, and it won't weigh down your backpack (or your Docker image).

---

## 🤔 Why Fence?

**The short answer:** By accident.

**The slightly longer answer:** LangChain used to be (is?) a pretty big package with a ton of dependencies. Great for PoCs, but in production? Not so much.

### The problems we faced:

- **🐘 It's BIG.** Takes up serious space (problematic in Lambda, containers, edge environments)
- **🌀 It's COMPLEX.** Overwhelming for new users, hard to debug in production
- **💥 It BREAKS.** Frequent breaking changes, version jumps that made us cry

As a result, many developers (especially those in large production environments) started building lightweight, custom solutions that favor **stability** and **robustness** over feature bloat.

### Enter Fence 🤺

We started building basic components from scratch for our Bedrock-heavy production environment. First came the `Link` class (_wink wink_), then templates, then agents... and before we knew it, we had a miniature package that was actually _fun_ to use.

**Fence strikes the perfect balance between convenience and flexibility.**

> **Note:** Fence isn't trying to replace LangChain for complex PoCs. But if you want a simple, lightweight, production-ready package that's easy to understand and extend, you're in the right place.

---

## 📦 Installation

```bash
pip install fence-llm
```

That's it. Seriously. No 500MB of transitive dependencies.

---

## 🚀 Quick Start

### Hello World (The Obligatory Example)

```python
from fence.links import Link
from fence.templates.string import StringTemplate
from fence.models.openai import GPT4omini

# Create a link
link = Link(
    model=GPT4omini(),
    template=StringTemplate("Write a haiku about {topic}"),
    name='haiku_generator'
)

# Run it
output = link.run(topic='fencing')['state']
print(output)
```

**Output:**

```
[2024-10-04 17:45:15] [ℹ️ INFO] [links.run:203]              Executing <haiku_generator> Link
Blades flash in the light,
En garde, the dance begins now,
Touch—victory's mine.
```

Much wow. Very poetry. 🎭

---

## 💪 What Can Fence Do?

Fence is built around a few core concepts that work together beautifully:

### 🤖 **Multi-Provider LLM Support**

Uniform interface across AWS Bedrock (Claude, Nova), OpenAI (GPT-4o), Anthropic, Google Gemini, Ollama, and Mistral. Switch models with a single line change.

👉 **[See all supported models →](docs/MODELS.md)**

### 🔗 **Links & Chains**

Composable building blocks that combine models, templates, and parsers. Chain them together for complex workflows.

👉 **[Learn about Links & Chains →](docs/LINKS_AND_CHAINS.md)**

### 🤖 **Agentic Workflows** ⭐

The crown jewel! Production-ready agents using the ReAct pattern:

- **`Agent`** - Classic ReAct with tool use and multi-level delegation
- **`BedrockAgent`** - Native Bedrock tool calling with streaming
- **`ChatAgent`** - Conversational agents for multi-agent systems

👉 **[Dive into Agents →](docs/AGENTS.md)**

### 🔌 **MCP Integration**

First-class support for the Model Context Protocol. Connect to MCP servers and automatically expose their tools to your agents.

👉 **[Explore MCP Integration →](docs/MCP.md)**

### 🎭 **Multi-Agent Systems**

Build collaborative agent systems with `RoundTable` where multiple agents discuss and solve problems together.

👉 **[Build Multi-Agent Systems →](docs/MULTI_AGENT.md)**

### 🧠 **Memory Systems**

Persistent and ephemeral memory backends (DynamoDB, SQLite, in-memory) for stateful conversations.

👉 **[Configure Memory →](docs/MEMORY.md)**

### 🛠️ **Tools & Utilities**

Custom tool creation, built-in tools, retry logic, parallelization, output parsers, logging callbacks, and benchmarking.

👉 **[Explore Tools & Utilities →](docs/TOOLS_AND_UTILITIES.md)**

---

## 📚 Documentation

- **[Models](docs/MODELS.md)** - All supported LLM providers and how to use them
- **[Links & Chains](docs/LINKS_AND_CHAINS.md)** - Building blocks for LLM workflows
- **[Agents](docs/AGENTS.md)** - ReAct agents, tool use, and delegation
- **[MCP Integration](docs/MCP.md)** - Model Context Protocol support
- **[Multi-Agent Systems](docs/MULTI_AGENT.md)** - RoundTable and collaborative agents
- **[Memory](docs/MEMORY.md)** - Persistent and ephemeral memory backends
- **[Tools & Utilities](docs/TOOLS_AND_UTILITIES.md)** - Custom tools, parsers, and helpers

---

## 🎯 Examples

### Simple Agent with Tools

```python
from fence.agents import Agent
from fence.models.openai import GPT4omini
from fence.tools.math import CalculatorTool

agent = Agent(
    identifier="math_wizard",
    model=GPT4omini(source="demo"),
    tools=[CalculatorTool()],
)

result = agent.run("What is 1337 * 42 + 999?")
print(result)  # Agent thinks, uses calculator, and answers!
```

### BedrockAgent with MCP

```python
from fence.agents.bedrock import BedrockAgent
from fence.mcp.client import MCPClient
from fence.models.bedrock import Claude37Sonnet

# Connect to MCP server
mcp_client = MCPClient(
    transport_type="streamable_http",
    url="https://your-mcp-server.com/mcp"
)

# Create agent with MCP tools
agent = BedrockAgent(
    identifier="mcp_agent",
    model=Claude37Sonnet(region="us-east-1"),
    mcp_clients=[mcp_client],  # Tools auto-registered!
)

result = agent.run("Search for customer data")
```

### Multi-Agent Collaboration

```python
from fence.troupe import RoundTable
from fence.agents import ChatAgent
from fence.models.openai import GPT4omini

# Create specialized agents
detective = ChatAgent(
    identifier="Detective",
    model=GPT4omini(source="roundtable"),
    profile="You are a sharp detective."
)

scientist = ChatAgent(
    identifier="Scientist",
    model=GPT4omini(source="roundtable"),
    profile="You are a forensic scientist."
)

# Let them collaborate
round_table = RoundTable(agents=[detective, scientist])
transcript = round_table.run(
    prompt="A painting was stolen. Let's investigate!",
    max_rounds=3
)
```

**More examples:**

- 📓 [Jupyter Notebooks](notebooks/) - Interactive tutorials
- 🎬 [Demo Scripts](demo/) - Runnable examples

---

## 🤝 Contributing

We welcome contributions! Whether it's:

- 🐛 Bug fixes
- ✨ New features (especially new model providers!)
- 📝 Documentation improvements
- 🧪 More tests
- 🎨 Better examples

Check out [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📄 License

MIT License - see [LICENSE.txt](LICENSE.txt) for details.

---

## 🙏 Acknowledgments

Inspired by LangChain, built for production, made with ❤️ by developers who got tired of dependency hell.

**Now go build something awesome! 🚀**
