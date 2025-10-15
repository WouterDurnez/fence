# 🔗 Links & Chains

Links and Chains are the fundamental building blocks of Fence. They let you compose LLM workflows in a clean, reusable way.

---

## Links

A **Link** combines three things:

1. **Model** - The LLM to use
2. **Template** - The prompt template
3. **Parser** (optional) - How to parse the output

Think of Links as reusable LLM components that you can chain together.

### Basic Link

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
result = link.run(topic='fencing')
print(result['state'])
```

### Link with Parser

```python
from fence.links import Link
from fence.templates.string import StringTemplate
from fence.parsers import BoolParser
from fence.models.bedrock import ClaudeHaiku

# Create a link that returns a boolean
sentiment_link = Link(
    model=ClaudeHaiku(),
    template=StringTemplate("Is this text positive? Answer true or false: {text}"),
    parser=BoolParser(),
    name='sentiment_analyzer'
)

# Run it
result = sentiment_link.run(text="I love this product!")
print(result['state'])  # True
```

### Link Configuration

```python
link = Link(
    model=GPT4omini(),
    template=StringTemplate("Summarize: {text}"),
    parser=None,                    # Optional parser
    name='summarizer',              # Link name for logging
    callbacks=[my_callback],        # Optional callbacks
    metadata={'version': '1.0'}     # Optional metadata
)
```

---

## Templates

Templates define how to format prompts. Fence supports multiple template types.

### StringTemplate

Simple string templates with variable substitution:

```python
from fence.templates.string import StringTemplate

template = StringTemplate("Write a {style} story about {topic}")

# Render with variables
prompt = template.render(style="sci-fi", topic="robots")
# "Write a sci-fi story about robots"
```

### MessagesTemplate

Structured messages for chat models:

```python
from fence.templates.messages import MessagesTemplate
from fence.templates.models import Messages, Message

# Create a messages template
messages = Messages(
    system="You are a {role}",
    messages=[
        Message(role="user", content="Tell me about {topic}"),
    ]
)

template = MessagesTemplate(source=messages)

# Render with variables
rendered = template.render(role="historian", topic="ancient Rome")
```

### Multi-Turn Conversations

```python
from fence.templates.models import Messages, Message

messages = Messages(
    system="You are a helpful assistant",
    messages=[
        Message(role="user", content="What is AI?"),
        Message(role="assistant", content="AI stands for Artificial Intelligence..."),
        Message(role="user", content="Tell me more about {aspect}"),
    ]
)

template = MessagesTemplate(source=messages)
rendered = template.render(aspect="machine learning")
```

---

## Parsers

Parsers extract structured data from LLM outputs.

### BoolParser

Extract boolean values:

```python
from fence.parsers import BoolParser

parser = BoolParser()

# Parse various formats
parser.parse("The answer is: true")  # True
parser.parse("False, because...")    # False
parser.parse("TRUE")                 # True
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
````

"""

data = parser.parse(output)

# {'name': 'Fence', 'version': '2.0.4', 'features': ['agents', 'mcp', 'streaming']}

````

---

## Chains

**Chains** connect multiple Links together, passing outputs from one to the next.

### Basic Chain

```python
from fence.chains import Chain
from fence.links import Link
from fence.templates.string import StringTemplate
from fence.models.bedrock import ClaudeHaiku

# Create links
topic_generator = Link(
    model=ClaudeHaiku(),
    template=StringTemplate("Generate a random topic for a {genre} story"),
    name="topic_generator"
)

story_writer = Link(
    model=ClaudeHaiku(),
    template=StringTemplate("Write a short {genre} story about: {topic}"),
    name="story_writer"
)

# Chain them together
chain = Chain(links=[topic_generator, story_writer])

# Run the chain
result = chain.run(genre="sci-fi")
print(result['state'])
````

**What happens:**

1. `topic_generator` runs with `genre="sci-fi"`
2. Output becomes `topic` variable
3. `story_writer` runs with `genre="sci-fi"` and `topic=<generated topic>`
4. Final output is the story

### Multi-Step Chain

```python
# Step 1: Generate outline
outliner = Link(
    model=ClaudeHaiku(),
    template=StringTemplate("Create a 3-point outline for an essay about {topic}"),
    name="outliner"
)

# Step 2: Expand each point
expander = Link(
    model=ClaudeHaiku(),
    template=StringTemplate("Expand this outline into a full essay:\n{outline}"),
    name="expander"
)

# Step 3: Add conclusion
concluder = Link(
    model=ClaudeHaiku(),
    template=StringTemplate("Add a strong conclusion to this essay:\n{essay}"),
    name="concluder"
)

# Chain them
essay_chain = Chain(links=[outliner, expander, concluder])

# Run
result = essay_chain.run(topic="renewable energy")
```

### Chain with Parsers

```python
# Extract structured data at each step
data_extractor = Link(
    model=GPT4omini(),
    template=StringTemplate("Extract key facts from: {text}"),
    parser=TOMLParser(),
    name="extractor"
)

summarizer = Link(
    model=GPT4omini(),
    template=StringTemplate("Summarize these facts: {facts}"),
    name="summarizer"
)

chain = Chain(links=[data_extractor, summarizer])
result = chain.run(text="Long article text...")
```

---

## Advanced Patterns

### Conditional Logic

```python
from fence.links import Link
from fence.parsers import BoolParser

# Check if content is appropriate
content_checker = Link(
    model=GPT4omini(),
    template=StringTemplate("Is this content appropriate? {text}"),
    parser=BoolParser(),
    name="checker"
)

result = content_checker.run(text="Some text")

if result['state']:
    # Process the content
    processor = Link(...)
    processed = processor.run(text="Some text")
else:
    # Handle inappropriate content
    print("Content flagged")
```

### Parallel Processing

```python
from fence.utils.optim import parallelize

# Create multiple links
links = [
    Link(model=GPT4omini(), template=StringTemplate("Analyze {text} for sentiment")),
    Link(model=GPT4omini(), template=StringTemplate("Extract entities from {text}")),
    Link(model=GPT4omini(), template=StringTemplate("Summarize {text}")),
]

# Run in parallel
results = parallelize(
    func=lambda link: link.run(text="Article text"),
    items=links,
    max_workers=3
)
```

### Retry Logic

```python
from fence.utils.optim import retry

# Wrap link execution with retry
@retry(max_attempts=3, delay=1.0)
def run_link_with_retry(link, **kwargs):
    return link.run(**kwargs)

result = run_link_with_retry(my_link, topic="AI")
```

---

## State Management

Chains maintain state across links:

```python
chain = Chain(links=[link1, link2, link3])

# Initial state
result = chain.run(topic="AI", style="formal")

# Access final state
print(result['state'])

# Access intermediate states
print(result['link1_output'])
print(result['link2_output'])
```

---

## Callbacks

Add callbacks to monitor link execution:

```python
def my_callback(link_name: str, input_data: dict, output: str):
    print(f"Link {link_name} completed")
    print(f"Input: {input_data}")
    print(f"Output: {output}")

link = Link(
    model=GPT4omini(),
    template=StringTemplate("Summarize {text}"),
    callbacks=[my_callback]
)

result = link.run(text="Long text...")
```

---

## Best Practices

### 1. Name Your Links

```python
# Good - easy to debug
link = Link(
    model=GPT4omini(),
    template=StringTemplate("..."),
    name="customer_sentiment_analyzer"
)

# Bad - hard to track
link = Link(
    model=GPT4omini(),
    template=StringTemplate("...")
)
```

### 2. Use Appropriate Models

```python
# Use cheaper models for simple tasks
simple_link = Link(
    model=ClaudeHaiku(),  # Fast and cheap
    template=StringTemplate("Extract the date from: {text}")
)

# Use powerful models for complex tasks
complex_link = Link(
    model=Claude35Sonnet(),  # More capable
    template=StringTemplate("Analyze the legal implications of: {contract}")
)
```

### 3. Add Parsers for Structured Output

```python
# Without parser - returns string
link1 = Link(
    model=GPT4omini(),
    template=StringTemplate("Is this spam? {email}")
)

# With parser - returns boolean
link2 = Link(
    model=GPT4omini(),
    template=StringTemplate("Is this spam? Answer true or false: {email}"),
    parser=BoolParser()
)
```

### 4. Keep Chains Focused

```python
# Good - focused chain
email_chain = Chain(links=[
    spam_detector,
    sentiment_analyzer,
    priority_classifier
])

# Bad - too many unrelated steps
everything_chain = Chain(links=[
    spam_detector,
    weather_fetcher,
    stock_analyzer,
    recipe_generator
])
```

---

## Real-World Example

```python
from fence.chains import Chain
from fence.links import Link
from fence.templates.string import StringTemplate
from fence.parsers import TOMLParser, BoolParser
from fence.models.bedrock import ClaudeHaiku, Claude35Sonnet

# Step 1: Extract customer info
info_extractor = Link(
    model=ClaudeHaiku(),
    template=StringTemplate("""
    Extract customer information from this email:
    {email}

    Return in TOML format:
    name = "..."
    email = "..."
    issue = "..."
    """),
    parser=TOMLParser(),
    name="info_extractor"
)

# Step 2: Check if urgent
urgency_checker = Link(
    model=ClaudeHaiku(),
    template=StringTemplate("""
    Is this issue urgent? Answer true or false.
    Issue: {issue}
    """),
    parser=BoolParser(),
    name="urgency_checker"
)

# Step 3: Generate response
response_generator = Link(
    model=Claude35Sonnet(),
    template=StringTemplate("""
    Generate a professional response to this customer:
    Name: {name}
    Issue: {issue}
    Urgent: {urgent}
    """),
    name="response_generator"
)

# Create the chain
support_chain = Chain(links=[
    info_extractor,
    urgency_checker,
    response_generator
])

# Process customer email
result = support_chain.run(email="Customer email text...")
print(result['state'])  # Final response
```

---

## Next Steps

- **[Build Agents →](AGENTS.md)**
- **[Explore Models →](MODELS.md)**
- **[Learn about Tools →](TOOLS_AND_UTILITIES.md)**
