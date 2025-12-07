# 🤖 Models

Fence provides a **uniform interface** across multiple LLM providers. No more wrestling with different APIs—just swap the model class and you're good to go!

---

## Supported Providers

### AWS Bedrock (Our Bread and Butter)

Bedrock models are the primary focus of Fence, with full support for streaming, tool calling, and cross-region inference.

#### Claude Models

```python
from fence.models.bedrock import (
    ClaudeInstant,
    ClaudeHaiku,
    ClaudeSonnet,
    Claude35Sonnet,
    Claude37Sonnet,
    Claude4Sonnet,
    Claude4Opus,
)

# Use any Claude model
model = Claude35Sonnet(
    source="my_app",
    region="us-east-1",
    cross_region="us"  # Optional: use cross-region inference
)

response = model("Explain quantum computing in simple terms")
```

**Available Claude Models:**

- `ClaudeInstant` - Fast, cost-effective
- `ClaudeHaiku` - Balanced performance
- `ClaudeSonnet` - Claude 3 Sonnet
- `Claude35Sonnet` - Claude 3.5 Sonnet (recommended)
- `Claude37Sonnet` - Claude 3.7 Sonnet
- `Claude4Sonnet` - Claude 4 Sonnet
- `Claude4Opus` - Claude 4 Opus (most capable)

#### Nova Models

```python
from fence.models.bedrock import NovaLite, NovaMicro, NovaPro

# Amazon's Nova models
model = NovaPro(
    source="my_app",
    cross_region="eu"
)

response = model("What's the weather like?")
```

**Available Nova Models:**

- `NovaMicro` - Smallest, fastest
- `NovaLite` - Balanced
- `NovaPro` - Most capable

#### Bedrock Configuration

```python
from fence.models.bedrock import Claude35Sonnet

model = Claude35Sonnet(
    source="my_feature",
    region="us-east-1",
    cross_region="us",  # Cross-region inference
    inferenceConfig={
        "temperature": 0.7,
        "maxTokens": 2000,
        "topP": 0.9,
    },
    full_response=False  # Return just the text, not full API response
)
```

---

### OpenAI

```python
from fence.models.openai import GPT4o, GPT4omini, GPT4

# Use OpenAI models
model = GPT4omini(source="my_app")

response = model("Write a poem about AI")
```

**Available Models:**

- `GPT4o` - GPT-4o (latest)
- `GPT4omini` - GPT-4o-mini (cost-effective)
- `GPT4` - GPT-4

**Configuration:**

```python
model = GPT4o(
    source="my_app",
    temperature=0.7,
    max_tokens=1000,
    top_p=0.9
)
```

Set your API key via environment variable:

```bash
export OPENAI_API_KEY="your-api-key"
```

---

### Anthropic (Direct API)

```python
from fence.models.anthropic import (
    Claude35Haiku,
    Claude35Sonnet,
    Claude37Sonnet,
    Claude4Sonnet,
    Claude4Opus,
)

# Use Anthropic's API directly
model = Claude35Sonnet()

response = model("Explain the theory of relativity")
```

**Available Models:**

- `Claude35Haiku` - Fast and efficient
- `Claude35Sonnet` - Balanced
- `Claude37Sonnet` - Claude 3.7
- `Claude4Sonnet` - Claude 4 Sonnet
- `Claude4Opus` - Most capable

Set your API key:

```bash
export ANTHROPIC_API_KEY="your-api-key"
```

---

### Google Gemini

```python
from fence.models.gemini import (
    Gemini1_5_Pro,
    GeminiFlash1_5,
    GeminiFlash2_0,
    Gemini2_5_Pro,
    Gemini2_5_Flash,
    Gemini2_5_FlashLite,
    Gemini  # Generic interface
)

# Use Gemini models
model = GeminiFlash2_0(source="my_app")

response = model("What are the benefits of renewable energy?")
```

**Available Models:**

- `Gemini1_5_Pro` - Most capable
- `GeminiFlash1_5` - Fast, efficient
- `GeminiFlash2_0` - Latest Flash model
- `Gemini2_5_Pro` - Gemini 2.5 Pro (experimental)
- `Gemini2_5_Flash` - Gemini 2.5 Flash (experimental)
- `Gemini2_5_FlashLite` - Gemini 2.5 Flash-Lite (experimental)
- `Gemini` - Generic interface for any Gemini model

**Configuration:**

```python
model = GeminiFlash2_0(
    source="my_app",
    temperature=0.8,
    max_tokens=1500
)
```

Set your API key:

```bash
export GOOGLE_API_KEY="your-api-key"
```

---

### Ollama (Local Models)

Run models locally with Ollama!

```python
from fence.models.ollama import Llama3_1, DeepSeekR1, Ollama

# Use pre-configured models
model = Llama3_1(source="local")

# Or use any Ollama model
model = Ollama(
    model_id="llama3.1",
    source="local",
    auto_pull=True  # Automatically pull if not available
)

response = model("Tell me a joke")
```

**Pre-configured Models:**

- `Llama3_1` - Meta's Llama 3.1
- `DeepSeekR1` - DeepSeek R1

**Generic Interface:**

```python
# Use any Ollama-compatible model
model = Ollama(
    model_id="mistral",
    source="local",
    base_url="http://localhost:11434"  # Default Ollama URL
)
```

---

### Mistral AI

```python
from fence.models.mistral import Mistral

# Use Mistral models
model = Mistral(
    model_id="mistral-large-latest",
    temperature=0.7,
    max_tokens=1000
)

response = model("Explain machine learning")
```

Set your API key:

```bash
export MISTRAL_API_KEY="your-api-key"
```

---

## Uniform Interface

All models share the same interface, making it easy to switch between providers:

```python
# All of these work the same way!
from fence.models.bedrock import Claude35Sonnet
from fence.models.openai import GPT4o
from fence.models.gemini import GeminiFlash2_0

models = [
    Claude35Sonnet(source="comparison"),
    GPT4o(source="comparison"),
    GeminiFlash2_0(source="comparison")
]

prompt = "What is the capital of France?"

for model in models:
    response = model(prompt)  # Same interface!
    print(f"{model.model_name}: {response}")
```

---

## Advanced Features

### Streaming (Bedrock Only)

```python
from fence.models.bedrock import Claude35Sonnet

model = Claude35Sonnet(source="streaming")

# Stream responses token by token
for chunk in model.stream("Write a long story about a dragon"):
    print(chunk, end='', flush=True)
```

### Message Templates

Use structured messages across all providers:

```python
from fence.templates.models import Messages, Message

messages = Messages(
    system="You are a helpful assistant",
    messages=[
        Message(role="user", content="What is AI?"),
        Message(role="assistant", content="AI stands for..."),
        Message(role="user", content="Tell me more")
    ]
)

# Works with any model!
response = model.invoke(messages)
```

### Logging & Metrics

```python
from fence.models.base import register_log_callback, register_log_tags

# Register custom logging
def my_logger(metrics: dict, tags: dict):
    print(f"Tokens: {metrics.get('output_tokens')}")
    print(f"Latency: {metrics.get('latency_ms')}ms")

register_log_callback(my_logger)
register_log_tags({"team": "ai", "env": "prod"})

# All model calls will now log metrics
model = Claude35Sonnet(source="app")
model("Hello!")  # Logs token usage, latency, etc.
```

---

## Model Selection Guide

| Use Case              | Recommended Model          | Why                                                |
| --------------------- | -------------------------- | -------------------------------------------------- |
| **Production (AWS)**  | `Claude35Sonnet`           | Best balance of performance, cost, and reliability |
| **Cost-Effective**    | `ClaudeHaiku`, `GPT4omini` | Fast and cheap for simple tasks                    |
| **Most Capable**      | `Claude4Opus`, `GPT4o`     | Complex reasoning and analysis                     |
| **Local Development** | `Llama3_1` (Ollama)        | No API costs, full privacy                         |
| **Fast Prototyping**  | `GeminiFlash2_0`           | Quick responses, good quality                      |
| **Tool Calling**      | `Claude35Sonnet` (Bedrock) | Native tool support with streaming                 |

---

## Next Steps

- **[Learn about Links & Chains →](LINKS_AND_CHAINS.md)**
- **[Build Agents →](AGENTS.md)**
- **[Explore Tools →](TOOLS_AND_UTILITIES.md)**
