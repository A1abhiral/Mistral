import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# Setup
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(env_path)

INDEX_NAME = "fyp-documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --------------------------------------------------
# Initialize vectorstore ONCE
# --------------------------------------------------
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

vectorstore = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embeddings
)

# --------------------------------------------------
# Retrieval function (frontend-safe)
# --------------------------------------------------
def retrieve_book_content(query: str, k: int = 3):
    docs = vectorstore.max_marginal_relevance_search(
        query=query,
        k=k,
        fetch_k=20
    )

    results = []
    for doc in docs:
        results.append({
            "text": doc.page_content,
            "source": os.path.basename(doc.metadata.get("source", "")),
            "page": int(doc.metadata.get("page", -1)) + 1
        })

    return results
