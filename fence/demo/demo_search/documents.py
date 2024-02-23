"""
Use this script to read a directory of PDF documents and save them to a vector store. This vector store can then be used
to define a retriever for the LLM.
"""


from dotenv import load_dotenv
from langchain.embeddings.bedrock import BedrockEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from fence.demo.demo_faq.utils import TextChunker
from fence.src.utils.base import DATA_DIR, setup_logging

load_dotenv()

VECTOR_DB_PATH = DATA_DIR / "search" / "paper_db"
PAPER_DB_PATH = DATA_DIR / "search" / "papers"

CHUNK_SIZE = 10_000

# Set up rich logging
logger = setup_logging()

# Load the documents
logger.info("Loading documents...")
documents = []

# Get all files from papers_path directory in a list
file_paths = list(PAPER_DB_PATH.glob("*.pdf"))

for file_path in file_paths:
    try:
        doc = PyPDFLoader(
            file_path=str(file_path),
        ).load()[0]
        documents.append(doc)
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")

# Chunk the documents
text_chunker = TextChunker(chunk_size=CHUNK_SIZE, overlap=0.1)
documents_chunked = []
for doc in documents:
    chunks = text_chunker.split_text(text=doc.page_content)

    # New document for chunks with same metadata
    documents_chunked.extend(
        [Document(metadata=doc.metadata, page_content=chunk) for chunk in chunks]
    )

# # MMA documents
# documents = [
#     Document(page_content="MMA is a combat sport that combines techniques from a variety of martial arts, including boxing, wrestling, and jiu-jitsu.", metadata={"source": "MMA doc 1"}),
#     Document(page_content="The sport of MMA has been growing in popularity since the 1990s. It grew out of the Brazilian Vale Tudo fighting style.", metadata={"source": "MMA doc 2"}),
#     Document(page_content="Boxing is one of the foundational components of MMA. Fighters use striking techniques such as jabs, hooks, and uppercuts to score points or knockout their opponents.", metadata={"source": "MMA doc 3"}),
#     Document(page_content="Wrestling plays a crucial role in MMA, emphasizing takedowns, throws, and ground control. Many successful MMA fighters have a strong background in wrestling.", metadata={"source": "MMA doc 4"}),
#     Document(page_content="Jiu-jitsu, particularly Brazilian Jiu-Jitsu (BJJ), is essential in MMA, focusing on ground fighting and submissions. Fighters use various holds and locks to force their opponents to submit.", metadata={"source": "MMA doc 5"}),
#     Document(page_content="Mixed Martial Arts competitions take place inside an octagonal cage known as the UFC Octagon. This unique structure provides a contained space for fighters to engage in both striking and grappling exchanges.", metadata={"source": "MMA doc 6"}),
#     Document(page_content="The Ultimate Fighting Championship (UFC) is the most prominent organization in the world of MMA. Established in 1993, it has played a pivotal role in popularizing the sport globally.", metadata={"source": "MMA doc 7"}),
#     Document(page_content="MMA fighters often follow strict training regimens that include cardiovascular conditioning, strength training, and skill-specific drills. The diverse skill set required in MMA demands comprehensive training.", metadata={"source": "MMA doc 8"}),
#     Document(page_content="In addition to physical prowess, mental toughness is crucial in MMA. Fighters must navigate the psychological challenges of the sport, including handling pressure, staying focused, and adapting to opponents' strategies.", metadata={"source": "MMA doc 9"}),
#     Document(page_content="The growth of MMA has led to the emergence of various weight classes, ensuring fair competition. Fighters compete in divisions ranging from flyweight to heavyweight, each with its own set of weight limits.", metadata={"source": "MMA doc 10"}),
#     Document(page_content="MMA events draw a diverse fan base, and the sport has transcended cultural boundaries. Fans appreciate the athleticism, skill, and strategy displayed by fighters in the cage.", metadata={"source": "MMA doc 11"}),
#     Document(page_content="While MMA has faced criticism for its perceived brutality, proponents argue that it is a legitimate sport that showcases the evolution of martial arts and provides athletes with a platform to showcase their skills.", metadata={"source": "MMA doc 12"}),
# ]
#
# # Various documents
# documents.extend([
# Document(page_content="Artificial Intelligence (AI) is a rapidly advancing field that involves creating intelligent machines capable of performing tasks that typically require human intelligence. Applications include machine learning, natural language processing, and computer vision.", metadata={"source": "AI doc 1"}),
#     Document(page_content="Climate change is a global environmental issue characterized by rising temperatures, extreme weather events, and disruptions to ecosystems. Mitigating climate change involves reducing greenhouse gas emissions and adopting sustainable practices.", metadata={"source": "Climate Change doc 1"}),
#     Document(page_content="Space exploration explores the universe beyond Earth, involving the study of celestial bodies, space missions, and the search for extraterrestrial life. Organizations like NASA play a pivotal role in advancing our understanding of the cosmos.", metadata={"source": "Space Exploration doc 1"}),
#     Document(page_content="Cuisine is a diverse and culturally significant aspect of human life. Different regions have unique culinary traditions, using local ingredients and cooking methods. Exploring global cuisines allows for a rich appreciation of cultural diversity.", metadata={"source": "Cuisine doc 1"}),
#     Document(page_content="Renewable energy sources, such as solar and wind power, are essential in addressing the world's energy needs while reducing reliance on fossil fuels. Transitioning to renewable energy is a key strategy for sustainable development.", metadata={"source": "Renewable Energy doc 1"}),
#
# ])
#
# # Contradicting documents
# documents.extend(
#     [
#         Document(
#             page_content="The Earth is flat and rests on the back of a giant turtle.",
#             metadata={"source": "Flat Earth doc 1"},
#         ),
#         Document(
#             page_content="The Earth is an oblate spheroid, meaning it is mostly spherical but slightly flattened at the poles.",
#             metadata={"source": "Flat Earth doc 2"},
#         ),
#     ]
# )

# Create the vector store
logger.info("Creating vector store...")
embeddings = BedrockEmbeddings()
docsearch = Chroma.from_documents(
    documents,
    embeddings,
    persist_directory=str(VECTOR_DB_PATH),
)

# Save the vector store
logger.info("Saving vector store...")
docsearch.persist()
