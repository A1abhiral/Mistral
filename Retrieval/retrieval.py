import os
import glob
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# --------------------------------------------------
# Load environment variables
# --------------------------------------------------
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in environment variables")

# --------------------------------------------------
# Configuration
# --------------------------------------------------
PDF_DIR = "Retrieval/data"
INDEX_NAME = "fyp-documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
EMBEDDING_DIMENSION = 384

MIN_CHUNK_LENGTH = 120  # remove headers/footers

BLACKLIST_KEYWORDS = [
    "contents",
    "table of contents",
    "answers",
    "answer key",
    "principles of physics - ii",
    "exercise",
    "multiple choice",
]

# --------------------------------------------------
# Load PDFs
# --------------------------------------------------
pdf_files = glob.glob(os.path.join(PDF_DIR, "*.pdf"))

if not pdf_files:
    raise ValueError(f"No PDF files found in {PDF_DIR}")

documents = []
for pdf_path in pdf_files:
    print(f"Loading: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents.extend(loader.load())

print(f"Loaded {len(documents)} pages")

# --------------------------------------------------
# Chunk documents
# --------------------------------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

raw_chunks = text_splitter.split_documents(documents)
print(f"Generated {len(raw_chunks)} raw chunks")

# --------------------------------------------------
# Clean chunks (CRITICAL)
# --------------------------------------------------
clean_chunks = []

for doc in raw_chunks:
    text = doc.page_content.strip()

    # Remove short junk chunks
    if len(text) < MIN_CHUNK_LENGTH:
        continue

    # Remove blacklisted content
    if any(bad in text.lower() for bad in BLACKLIST_KEYWORDS):
        continue

    # Optional: normalize whitespace
    doc.page_content = " ".join(text.split())

    clean_chunks.append(doc)

print(f"Clean chunks after filtering: {len(clean_chunks)}")

# --------------------------------------------------
# Initialize embeddings
# --------------------------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

# --------------------------------------------------
# Initialize Pinecone
# --------------------------------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)

existing_indexes = [idx.name for idx in pc.list_indexes()]

if INDEX_NAME not in existing_indexes:
    print(f"Creating Pinecone index: {INDEX_NAME}")
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIMENSION,
        metric="cosine",
        spec={
            "serverless": {
                "cloud": "aws",
                "region": "us-east-1"
            }
        }
    )
else:
    print(f"Using existing Pinecone index: {INDEX_NAME}")

# --------------------------------------------------
# Upload to Pinecone
# --------------------------------------------------
print("Uploading clean chunks to Pinecone...")

vectorstore = PineconeVectorStore.from_documents(
    documents=clean_chunks,
    embedding=embeddings,
    index_name=INDEX_NAME
)

print("✅ Improved ingestion completed successfully!")
