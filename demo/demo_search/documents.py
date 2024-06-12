"""
Use this script to read a directory of PDF documents and save them to a vector store. This vector store can then be used
to define a retriever for the LLM.
"""

from dotenv import load_dotenv
from langchain.embeddings.bedrock import BedrockEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from fence.utils.nlp import TextChunker
from fence.utils.base import DATA_DIR
from fence.utils.logger import setup_logging

load_dotenv()

VECTOR_DB_PATH = DATA_DIR / "search" / "paper_db"
PAPER_DB_PATH = DATA_DIR / "search" / "papers"

CHUNK_SIZE = 20_000

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

# Add the chunked documents
documents.extend(documents_chunked)

# Add default object type 'document' to all documents
for doc in documents:
    if "object_type" not in doc.metadata:
        doc.metadata["object_type"] = "document"

# Create the vector store
logger.info("Creating vector store...")
embeddings = BedrockEmbeddings()
docsearch = Chroma.from_documents(
    documents,
    embeddings,
    persist_directory=str(VECTOR_DB_PATH),
)

docsearch.similarity_search("MMA", k=5)
