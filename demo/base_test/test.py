

from fence import Link, ClaudeHaiku, StringTemplate, setup_logging

logger = setup_logging(log_level="INFO", serious_mode=True)

# Create a link
model = ClaudeHaiku(source="test-haiku", region='us-east-1')
link = Link(
    name='meaning_of_link',
    llm=model,
    template=StringTemplate("What is the meaning of life? Be very {adjective}."),
)
result = link.run(adjective="pessimistic")
logger.critical(result)