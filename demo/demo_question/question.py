from fence import Link, ClaudeHaiku, StringTemplate, setup_logging
from time import time
logger = setup_logging(log_level="INFO", serious_mode=True)

# Create a link
start = time()
model = ClaudeHaiku(source="test-haiku", region="us-east-1")
link = Link(
    name="meaning_of_link",
    llm=model,
    template=StringTemplate("Is the phrase between triple backticks a question? Ignore question marks and focus on the semantics."
                            "```{input}```"
                            " Response with True or False only."),
)
result = link.run(input="watt is de eenheid van vermogen?")
logger.critical(result)
logger.critical(f"Time taken: {time() - start} seconds")
