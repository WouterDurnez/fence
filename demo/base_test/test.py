import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))



import logging
import logging.config
from fence import Link, ClaudeHaiku, StringTemplate
from fence.utils.logger import ColorFormatter

from pprint import pprint as pp

# Set log level
# logging.basicConfig(
#     format="%(asctime)s [%(levelname)s] [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
#     level='INFO')
import yaml

# Load the config file
from fence.conf.log_config import config_dict

# Configure the logging module with the config file
logging.config.dictConfig(config_dict)

logger = logging.getLogger(__name__)

logger.info("Hello, World!")

# Create a link
model = ClaudeHaiku(source="test-haiku", region='us-east-1')
link = Link(
    name='meaning_of_link',
    llm=model,
    template=StringTemplate("What is the meaning of life? Be very {adjective}."),
)
result = link.run(adjective="pessimistic")
logger.critical(result)