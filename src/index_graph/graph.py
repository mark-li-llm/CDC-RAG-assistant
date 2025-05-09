# src/index_graph/graph.py
from pathlib import Path
import os, faulthandler
faulthandler.enable()            # Help locating potential crashes

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models as qmodels
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# Load environment variables from .env file (using correct absolute path)
def load_env():
    """Load environment variables using absolute path to ensure correct loading"""
    # Get project root directory
    module_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root_dir = os.path.dirname(module_dir)
    env_path = os.path.join(root_dir, '.env')
    
    print(f"Attempting to load environment variables from: {env_path}")
    load_dotenv(env_path)
    
    # Debug information
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        key_preview = f"{openai_api_key[:5]}...{openai_api_key[-4:]}"
        print(f"Loaded OPENAI_API_KEY from {env_path}: {key_preview}")
    else:
        print(f"Warning: Failed to load OPENAI_API_KEY from {env_path}")

# Call the environment variable loading function
load_env()


DATA_DIR   = Path("data/txt")
COLLECTION = "articles_demo"
EMB_MODEL  = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")

def _load_docs() -> list[Document]:
    docs = []
    for fp in DATA_DIR.glob("*.txt"):
        lines = fp.read_text(encoding="utf-8").splitlines()
        body  = "\n".join(lines[3:])          # Skip title/date/separator line
        docs.append(Document(page_content=body, metadata={"file": fp.name}))
    return docs

def build_index(rebuild: bool = False):
    """
    Build vector index.
    
    Args:
        rebuild: Whether to force rebuild the index, default is False, meaning use existing index if it exists
    
    Returns:
        Dictionary containing the vector store
    """
    embeddings = OpenAIEmbeddings(model=EMB_MODEL)
    # ========== New: Cloud / Local Auto Selection ==========
    QDRANT_URL = os.getenv("QDRANT_URL")       # e.g. https://xxxx.cloud.qdrant.io
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

    if QDRANT_URL:   # ← If URL is configured, connect to cloud
        client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=30.0,           # Allow longer timeout for cloud network latency
        )
    else:            # Otherwise use local embedded
        client = QdrantClient(path="./qdrant_data")

    # Check if collection exists
    collections = [c.name for c in client.get_collections().collections]
    collection_exists = COLLECTION in collections
    
    # If collection doesn't exist or force rebuild, create/rebuild it
    if rebuild and collection_exists:
        print(f"🔄 Forcing index rebuild, deleting existing collection '{COLLECTION}'")
        client.delete_collection(collection_name=COLLECTION)
        collection_exists = False
    
    if not collection_exists:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=qmodels.VectorParams(
                size=embeddings.embed_query("test").shape[0],
                distance=qmodels.Distance.COSINE,
            ),
        )

    vs = QdrantVectorStore(
        client=client, collection_name=COLLECTION, embedding=embeddings
    )

    # Check if collection is empty or force rebuild
    if not collection_exists or client.count(COLLECTION).count == 0 or rebuild:
        print("🆕 Building Qdrant index...")
        # Use RecursiveCharacterTextSplitter instead of SemanticChunker
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        vs.add_documents(splitter.split_documents(_load_docs()))
    else:
        print("🔄 Qdrant index loaded")

    return {"vectorstore": vs}
