# 🛠️ Tools & Utilities

Fence comes with a rich set of tools and utilities for building production-ready LLM applications.

---

## Table of Contents

- [Agent Tools](#agent-tools)
- [Output Parsers](#output-parsers)
- [Retry Logic](#retry-logic)
- [Parallelization](#parallelization)
- [Logging & Callbacks](#logging--callbacks)
- [Benchmarking](#benchmarking)
- [Embeddings](#embeddings)

---

## Agent Tools

Tools extend agent capabilities by giving them access to external functions and APIs.

### Built-in Tools

#### CalculatorTool

Perform mathematical calculations:

```python
from fence.tools.math import CalculatorTool
from fence.agents import Agent
from fence.models.openai import GPT4omini

agent = Agent(
    identifier="math_agent",
    model=GPT4omini(source="app"),
    tools=[CalculatorTool()]
)

result = agent.run("What is 1337 * 42 + 999?")
# Agent uses calculator and returns: "57153"
```

#### PrimeTool

Check if numbers are prime:

```python
from fence.tools.math import PrimeTool

agent = Agent(
    identifier="math_agent",
    model=GPT4omini(source="app"),
    tools=[PrimeTool()]
)

result = agent.run("Is 97 a prime number?")
# Agent uses PrimeTool and returns: "Yes, 97 is a prime number"
```

#### TextInverterTool

Reverse text strings:

```python
from fence.tools.text import TextInverterTool

agent = Agent(
    identifier="text_agent",
    model=GPT4omini(source="app"),
    tools=[TextInverterTool()]
)

result = agent.run("Invert the text: Hello World")
# Agent returns: "dlroW olleH"
```

#### ScratchPadTool

Temporary storage for agents:

```python
from fence.tools.scratch import ScratchPadTool

agent = Agent(
    identifier="memory_agent",
    model=GPT4omini(source="app"),
    tools=[ScratchPadTool()]
)

result = agent.run("Remember that the secret code is 1234, then tell me what it is")
# Agent stores and retrieves from scratchpad
```

### Creating Custom Tools

Build your own tools by extending `BaseTool`:

```python
from fence.tools.base import BaseTool

class WeatherTool(BaseTool):
    """A tool that fetches weather information."""

    name = "get_weather"
    description = "Get current weather for a location"

    def __init__(self):
        super().__init__()
        self.parameters = {
            "location": {
                "type": "string",
                "description": "City name or zip code"
            },
            "units": {
                "type": "string",
                "description": "Temperature units (celsius or fahrenheit)",
                "enum": ["celsius", "fahrenheit"]
            }
        }

    def run(self, location: str, units: str = "celsius", **kwargs) -> str:
        """
        Fetch weather data.

        :param location: City name or zip code
        :param units: Temperature units
        :param kwargs: Additional arguments (includes 'environment')
        :return: Weather information
        """
        # Your implementation here
        # Access environment variables: kwargs.get('environment', {})

        return f"The weather in {location} is sunny and 72°{units[0].upper()}"

# Use with an agent
from fence.agents import Agent

agent = Agent(
    identifier="weather_agent",
    model=GPT4omini(source="app"),
    tools=[WeatherTool()]
)

result = agent.run("What's the weather in Paris?")
```

### Advanced Tool Example

```python
import requests
from fence.tools.base import BaseTool

class APITool(BaseTool):
    """Tool that calls an external API."""

    name = "search_database"
    description = "Search the customer database"

    def __init__(self):
        super().__init__()
        self.parameters = {
            "query": {
                "type": "string",
                "description": "Search query"
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results",
                "default": 10
            }
        }

    def run(self, query: str, limit: int = 10, **kwargs) -> str:
        """
        Search the database.

        :param query: Search query
        :param limit: Max results
        :param kwargs: Additional arguments
        :return: Search results
        """
        # Get environment variables
        env = kwargs.get('environment', {})
        api_key = env.get('api_key')
        base_url = env.get('base_url')

        # Make API call
        response = requests.get(
            f"{base_url}/search",
            params={"q": query, "limit": limit},
            headers={"Authorization": f"Bearer {api_key}"}
        )

        if response.status_code == 200:
            results = response.json()
            return f"Found {len(results)} results: {results}"
        else:
            return f"Error: {response.status_code}"

# Use with environment variables
agent = Agent(
    identifier="api_agent",
    model=GPT4omini(source="app"),
    tools=[APITool()],
    environment={
        "api_key": "secret_key_123",
        "base_url": "https://api.example.com"
    }
)
```

---

## Output Parsers

Extract structured data from LLM outputs.

### BoolParser

Parse boolean values:

```python
from fence.parsers import BoolParser

parser = BoolParser()

# Various formats work
parser.parse("The answer is: true")     # True
parser.parse("False, because...")       # False
parser.parse("TRUE")                    # True
parser.parse("It's definitely false")   # False
```

### TripleBacktickParser

Extract code blocks:

````python
from fence.parsers import TripleBacktickParser

parser = TripleBacktickParser()

output = """
Here's the code:
```python
def hello():
    print("Hello!")
````

"""

code = parser.parse(output)

# "def hello():\n print(\"Hello!\")"

````

### TOMLParser

Parse TOML data:

```python
from fence.parsers import TOMLParser

parser = TOMLParser()

output = """
```toml
name = "Fence"
version = "2.0.4"
features = ["agents", "mcp", "streaming"]

[config]
debug = true
max_retries = 3
````

"""

data = parser.parse(output)

# {

# 'name': 'Fence',

# 'version': '2.0.4',

# 'features': ['agents', 'mcp', 'streaming'],

# 'config': {'debug': True, 'max_retries': 3}

# }

````

### Using Parsers with Links

```python
from fence.links import Link
from fence.parsers import BoolParser
from fence.templates.string import StringTemplate

link = Link(
    model=GPT4omini(),
    template=StringTemplate("Is this text positive? Answer true or false: {text}"),
    parser=BoolParser(),
    name="sentiment"
)

result = link.run(text="I love this!")
print(result['state'])  # True (boolean, not string)
````

---

## Retry Logic

Handle transient failures with automatic retries:

```python
from fence.utils.optim import retry

@retry(max_attempts=3, delay=1.0)
def flaky_api_call():
    """This function will retry up to 3 times with 1 second delay."""
    response = requests.get("https://api.example.com/data")
    response.raise_for_status()
    return response.json()

# Use it
try:
    data = flaky_api_call()
except Exception as e:
    print(f"Failed after 3 attempts: {e}")
```

### Retry with Exponential Backoff

```python
@retry(max_attempts=5, delay=1.0, backoff=2.0)
def api_call_with_backoff():
    """
    Retries with exponential backoff:
    - Attempt 1: immediate
    - Attempt 2: wait 1s
    - Attempt 3: wait 2s
    - Attempt 4: wait 4s
    - Attempt 5: wait 8s
    """
    return requests.get("https://api.example.com/data").json()
```

### Retry with LLM Calls

```python
from fence.models.openai import GPT4omini

model = GPT4omini(source="app")

@retry(max_attempts=3, delay=2.0)
def get_llm_response(prompt: str) -> str:
    """Retry LLM calls on failure."""
    return model(prompt)

response = get_llm_response("What is AI?")
```

---

## Parallelization

Run multiple operations concurrently:

```python
from fence.utils.optim import parallelize

# Process multiple prompts in parallel
prompts = [
    "Summarize: Article 1...",
    "Summarize: Article 2...",
    "Summarize: Article 3...",
]

results = parallelize(
    func=model.invoke,
    items=prompts,
    max_workers=3
)

for i, result in enumerate(results):
    print(f"Summary {i+1}: {result}")
```

### Parallel Link Execution

```python
from fence.links import Link

# Create multiple links
links = [
    Link(model=GPT4omini(), template=StringTemplate("Analyze sentiment: {text}")),
    Link(model=GPT4omini(), template=StringTemplate("Extract entities: {text}")),
    Link(model=GPT4omini(), template=StringTemplate("Summarize: {text}")),
]

# Run all links in parallel
results = parallelize(
    func=lambda link: link.run(text="Article text..."),
    items=links,
    max_workers=3
)
```

### Custom Parallel Processing

```python
def process_document(doc_id: str) -> dict:
    """Process a single document."""
    # Your processing logic
    return {"id": doc_id, "status": "processed"}

document_ids = ["doc1", "doc2", "doc3", "doc4", "doc5"]

results = parallelize(
    func=process_document,
    items=document_ids,
    max_workers=5
)
```

---

## Logging & Callbacks

Monitor and track LLM usage with callbacks:

### Register Global Callback

```python
from fence.models.base import register_log_callback, register_log_tags

def my_logger(metrics: dict, tags: dict):
    """Custom logging function."""
    print(f"Model: {tags.get('model_name')}")
    print(f"Tokens: {metrics.get('output_tokens')}")
    print(f"Latency: {metrics.get('latency_ms')}ms")
    print(f"Cost: ${metrics.get('cost', 0):.4f}")

# Register the callback
register_log_callback(my_logger)

# Register global tags
register_log_tags({
    "team": "ai",
    "env": "production",
    "project": "chatbot"
})

# All model calls will now log
model = GPT4omini(source="app")
model("Hello!")  # Triggers the callback
```

### Integration with Monitoring Systems

```python
import datadog

def datadog_logger(metrics: dict, tags: dict):
    """Send metrics to Datadog."""
    for metric_name, value in metrics.items():
        datadog.statsd.gauge(
            f"llm.{metric_name}",
            value,
            tags=[f"{k}:{v}" for k, v in tags.items()]
        )

register_log_callback(datadog_logger)
```

### Per-Model Configuration

```python
model = GPT4omini(
    source="feature_x",
    metric_prefix="chatbot",
    extra_tags={"version": "2.0", "feature": "summarization"}
)

# Metrics will be prefixed and tagged accordingly
```

---

## Benchmarking

Compare model performance:

```python
from fence.benchmark.run import run_benchmark
from fence.models.bedrock import Claude35Sonnet
from fence.models.openai import GPT4o
from fence.models.gemini import GeminiFlash2_0

# Define models to benchmark
models = [
    Claude35Sonnet(source="benchmark"),
    GPT4o(source="benchmark"),
    GeminiFlash2_0(source="benchmark")
]

# Define test prompts
prompts = [
    "Explain quantum computing in simple terms",
    "Write a haiku about AI",
    "Summarize the benefits of renewable energy"
]

# Run benchmark
results = run_benchmark(
    models=models,
    prompts=prompts,
    iterations=10  # Run each prompt 10 times
)

# Results include:
# - Average latency per model
# - Token usage
# - Cost estimates
# - Success rates
```

---

## Embeddings

Generate embeddings for semantic search and similarity:

### Bedrock Embeddings

```python
from fence.embeddings.bedrock import TitanEmbeddings

# Create embeddings model
embeddings = TitanEmbeddings(
    source="search",
    region="us-east-1"
)

# Generate embedding
text = "Fence is a lightweight LLM library"
embedding = embeddings.embed(text)

# embedding is a list of floats (vector)
print(f"Embedding dimension: {len(embedding)}")
```

### Configuration

```python
embeddings = TitanEmbeddings(
    source="app",
    region="us-east-1",
    dimensions=1024,      # Embedding dimensions
    normalize=True,       # Normalize vectors
    embeddingTypes=["float"]
)
```

---

## Utility Functions

### Time Measurement

```python
from fence.utils.base import time_it

@time_it
def slow_function():
    """This function's execution time will be logged."""
    # Your code here
    pass

slow_function()  # Logs: "slow_function took 1.23s"
```

### NLP Utilities

```python
from fence.utils.nlp import check_relevancy

# Check if text is relevant to a topic
is_relevant = check_relevancy(
    text="This article discusses machine learning algorithms",
    topic="artificial intelligence",
    model=GPT4omini(source="util")
)

print(is_relevant)  # True or False
```

---

## Best Practices

### 1. Use Retry for Production

```python
# Always wrap external calls
@retry(max_attempts=3, delay=1.0)
def production_llm_call(prompt: str) -> str:
    return model(prompt)
```

### 2. Parallelize When Possible

```python
# Process multiple items concurrently
results = parallelize(
    func=process_item,
    items=large_list,
    max_workers=10
)
```

### 3. Monitor with Callbacks

```python
# Track usage and costs
register_log_callback(send_to_monitoring_system)
register_log_tags({"env": "prod", "team": "ai"})
```

### 4. Use Appropriate Parsers

```python
# Get structured output
link = Link(
    model=model,
    template=template,
    parser=TOMLParser()  # Structured data
)
```

---

## Next Steps

- **[Learn about Agents →](AGENTS.md)**
- **[Explore Models →](MODELS.md)**
- **[Build Multi-Agent Systems →](MULTI_AGENT.md)**
