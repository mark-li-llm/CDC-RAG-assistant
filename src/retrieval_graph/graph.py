# src/retrieval_graph/graph.py
import os
import sys
from typing import Dict, List
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import SystemMessage, HumanMessage
# Import sentence_transformers for reranking
from sentence_transformers import CrossEncoder

# Add src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.insert(0, os.path.dirname(src_dir))  # Add project root directory

# Now using absolute import
from src.core.prompts import build_prompt

# Load environment variables from .env file (using correct absolute path)
def load_env():
    """Load environment variables using absolute path to ensure correct loading"""
    # Get project root directory
    module_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root_dir = os.path.dirname(module_dir)
    env_path = os.path.join(root_dir, '.env')
    
    print(f"Attempting to load environment variables from: {env_path}")
    # Force reload environment variables
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    try:
                        key, value = line.strip().split('=', 1)
                        os.environ[key] = value.strip('"').strip("'")
                    except ValueError:
                        continue
    
    # Debug information
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        key_preview = f"{openai_api_key[:5]}...{openai_api_key[-4:]}"
        print(f"Loaded OPENAI_API_KEY from {env_path}: {key_preview}")
    else:
        print(f"Warning: Failed to load OPENAI_API_KEY from {env_path}")
    
    # Output model name
    model_name = os.getenv("CHAT_MODEL", "")
    print(f"Actual model name used: {model_name}")

# Call the environment variable loading function
load_env()

TOP_K    = 15
FINAL_K  = 5
# Use complete version number for model name, based on test results
LLM_MODEL = os.getenv("CHAT_MODEL", "gpt-4.1-mini-2025-04-14")
print(f"LLM_MODEL value: {LLM_MODEL}")

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# —Hybrid Retrieval—
def _hybrid(inputs: Dict) -> Dict:
    q, vs = inputs["query"], inputs["vectorstore"]
    dense = vs.similarity_search(q, k=TOP_K)
    bm25  = BM25Retriever.from_documents(dense)
    # Use invoke() instead of the deprecated get_relevant_documents()
    sparse = bm25.invoke(q, config={"k": TOP_K})
    docs = {d.page_content: d for d in dense + sparse}.values()
    return {"docs": list(docs), "query": q}

# —Custom Cross-Encoder Reranking—
def _rerank(inputs: Dict) -> Dict:
    try:
        # Directly use sentence_transformers CrossEncoder for reranking
        docs = inputs["docs"]
        query = inputs["query"]
        
        # Create CrossEncoder model
        model = CrossEncoder('BAAI/bge-reranker-base')
        
        # Create document-query pairs for each document
        doc_query_pairs = [(query, doc.page_content) for doc in docs]
        
        # Calculate relevance scores
        scores = model.predict(doc_query_pairs)
        
        # Combine documents and scores, sort by score in descending order
        doc_score_pairs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        
        # Get the top FINAL_K documents
        best_docs = [doc for doc, _ in doc_score_pairs[:FINAL_K]]
        
        # To maintain interface consistency, create the same ctx and srcs return format
        ctx = "\n\n---\n\n".join(d.page_content for d in best_docs)
        
        # Basic source information
        srcs = [(d.metadata["file"], d.metadata.get("cid", 0)) for d in best_docs]
        
        # Detailed source information - including document content and metadata
        detailed_sources = []
        for doc, score in doc_score_pairs[:FINAL_K]:
            detailed_sources.append({
                "file": doc.metadata["file"],
                "chunk_id": doc.metadata.get("cid", 0),
                "content": doc.page_content,
                "score": float(score),
                "metadata": {k:v for k,v in doc.metadata.items() if k not in ["file", "cid"]}
            })
        
        return {
            "ctx": ctx, 
            "srcs": srcs, 
            "query": query,
            "detailed_sources": detailed_sources
        }
    except Exception as e:
        print(f"Error during reranking: {e}")
        # If reranking fails, use the initial documents
        docs = inputs["docs"][:FINAL_K]  # Simply take the first FINAL_K
        ctx = "\n\n---\n\n".join(d.page_content for d in docs)
        srcs = [(d.metadata["file"], d.metadata.get("cid", 0)) for d in docs]
        
        # Simple detailed source information
        detailed_sources = []
        for doc in docs:
            detailed_sources.append({
                "file": doc.metadata["file"],
                "chunk_id": doc.metadata.get("cid", 0),
                "content": doc.page_content,
                "score": 0.0,  # No score
                "metadata": {k:v for k,v in doc.metadata.items() if k not in ["file", "cid"]}
            })
            
        return {
            "ctx": ctx, 
            "srcs": srcs, 
            "query": inputs["query"],
            "detailed_sources": detailed_sources
        }

# —Generation—
def _generate(inputs: Dict) -> Dict:
    try:
        prompt = build_prompt("detailed", inputs["ctx"], inputs["query"])
        # Get model name from environment again
        model_name = os.getenv("CHAT_MODEL", "gpt-4.1-mini-2025-04-14")
        print(f"Generating answer using model: {model_name}")
        
        # Check API key type
        api_key = os.getenv("OPENAI_API_KEY", "")
        if api_key.startswith("sk-proj-"):
            print("Warning: You are using a project-specific API key (sk-proj-...), which may only access specific models")
        
        # Use OpenAI native SDK instead of langchain wrapper
        try:
            response = openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a CDC Weekly Reports assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )
            answer = response.choices[0].message.content.strip()
            return {
                "answer": answer, 
                "sources": inputs["srcs"],
                "detailed_sources": inputs.get("detailed_sources", [])
            }
        except Exception as model_error:
            print(f"Model access error: {model_error}")
            # Fallback solution not dependent on external API
            summary = "Due to API limitations, unable to generate response using OpenAI model. Here is a summary of the relevant content:"
            context_summary = inputs["ctx"][:500] + "..." if len(inputs["ctx"]) > 500 else inputs["ctx"]
            return {
                "answer": f"{summary}\n\n{context_summary}",
                "sources": inputs["srcs"],
                "detailed_sources": inputs.get("detailed_sources", [])
            }
    except Exception as e:
        print(f"Error generating answer: {e}")
        # If generation fails, return a friendly error message
        return {
            "answer": f"I'm sorry, I encountered a problem processing your query. This may be due to API limitations or connection issues. Error message: {str(e)}",
            "sources": inputs["srcs"],
            "detailed_sources": inputs.get("detailed_sources", [])
        }

def build_chain():
    return RunnableLambda(_hybrid) | RunnableLambda(_rerank) | RunnableLambda(_generate)
