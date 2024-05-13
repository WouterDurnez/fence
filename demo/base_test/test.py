import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from fence.utils.base import setup_logging
from fence import Link, ClaudeHaiku, StringTemplate

logger = setup_logging(__name__, serious_mode=False)

logger.info("Hello, World!")

# Create a link
model = ClaudeHaiku(source="test-haiku", region='us-east-1')
link = Link(
    llm=model,
    template=StringTemplate("What is the meaning of life? Be very {adjective}."),
)
result = link.run(adjective="pessimistic")
logger.critical(result)