# CDC Weekly Reports RAG Intelligent Question‑Answering System

<p align="center">
  <video src="demo.mp4" controls loop muted style="max-width:100%"></video>
</p>

![Demo Video](demo.gif)

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-Latest-green.svg)](https://github.com/langchain-ai/langchain)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.33+-red.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)

Built with **LangChain**, this **enterprise‑grade Retrieval‑Augmented Generation (RAG) QA system** is optimised for CDC Weekly Reports data. It features a modular pipeline and integrates hybrid retrieval with smart re‑ranking.

## 🎯 Project Highlights

### 🚀 **Core Technical Highlights**

* **Modular pipeline architecture**: Indexing + Retrieval + Research sub‑pipeline (LangChain)
* **Hybrid retrieval optimisation**: Dense Vector Search + BM25 Sparse Search + Cross‑Encoder re‑ranking
* **Asynchronous parallel processing**: Multi‑query parallel retrieval for faster responses
* **Cloud‑native deployment**: Qdrant Cloud + Docker containers
* **Enterprise‑grade scalability**: Supports multiple vector stores (Qdrant / Elasticsearch / Pinecone / MongoDB)

### 💡 **Intelligent QA Capabilities**

* **Domain expertise**: Fine‑tuned for CDC Weekly Reports
* **Context awareness**: Coherent answers using conversation history
* **Source tracing**: Detailed citations with confidence scores
* **Multi‑turn dialogue**: Handles complex interactive conversations

## 🏗️ System Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────┐
│  Streamlit  │────►│    LangChain     │────►│   Qdrant    │
│   Web UI    │◄────│   Orchestrator   │     │  Vector DB  │
└─────────────┘     └─────────┬────────┘     └─────────────┘
                              │
                    ┌─────────┼─────────┐
                    ▼         ▼         ▼
              ┌──────────┐  Nginx  ┌─────────┬─────────┐
              │ OpenAI   │   (LB)  │ vLLM #1 │ vLLM #2 │
              │ (Cloud)  │◄────────│(Server) │(Server) │
              └──────────┘         └─────────┴─────────┘
```

### LLM Backends
- Cloud path: direct calls to OpenAI API (default).
- Self‑hosted path: Nginx reverse proxy routes requests to two vLLM services (OpenAI‑compatible API). Switch by setting `OPENAI_BASE_URL` to the Nginx endpoint.

### 🔄 **Retrieval Workflow**

1. **Query understanding** → analyse user intent
2. **Query expansion** → generate related queries
3. **Parallel retrieval** → dense & sparse search in parallel
4. **Smart re‑ranking** → Cross‑Encoder semantic re‑ranking
5. **Answer generation** → craft professional answers
6. **Source attribution** → attach citations

## 🛠️ Tech Stack

| Layer               | Component               | Description                                     |
| ------------------- | ----------------------- | ----------------------------------------------- |
| **Frontend**        | Streamlit               | Modern web UI + custom CSS                      |
| **AI Framework**    | LangChain               | Runnables pipeline + async orchestration        |
| **LLM**             | OpenAI GPT‑4.1          | Answer generation + query understanding         |
| **LLM Gateway**     | Nginx                   | Reverse proxy, OpenAI‑compatible routing to vLLM|
| **Self‑Hosted LLM** | vLLM (x2 services)      | On‑server inference, OpenAI‑compatible API      |
| **Embedding Model** | text‑embedding‑ada‑002  | Document vectorisation                          |
| **Vector DB**       | Qdrant Cloud            | High‑performance vector storage & search        |
| **Re‑ranking**      | BAAI/bge‑reranker‑base  | Cross‑Encoder semantic re‑ranking               |
| **Retrieval**       | BM25 + Vector Search    | Hybrid retrieval strategy                       |
| **Deployment**      | Docker + Docker Compose | Containerised deployment                        |

## 🚀 Quick Start

### Method 1: Docker Deployment (recommended)

1. **Clone the project**

```bash
git clone <your-repo-url>
cd rag_0508
```

2. **Copy and edit environment variables**

```bash
cp .env.example .env
```

Fill in your keys (at minimum `OPENAI_API_KEY`, and `QDRANT_URL`/`QDRANT_API_KEY` when using Qdrant Cloud).

3. **Choose runtime mode**


*Cloud Qdrant (default)* — keep `COMPOSE_PROFILES=cloud` in `.env`.

*Local embedded Qdrant* — set `COMPOSE_PROFILES=local` in `.env` (or export it) and ensure `qdrant_data/` exists on the host.

> Use self‑hosted LLM via Nginx: point the app to your Nginx OpenAI‑compatible endpoint by setting `OPENAI_BASE_URL` (e.g., `http://<nginx-host>:8081/v1`) and set `CHAT_MODEL` to the model name loaded by vLLM. Leave `OPENAI_BASE_URL` empty to use OpenAI cloud.

4. **Build and run the container**

```bash
# Cloud profile (default)
make docker-up
# or
docker compose --profile cloud up -d --build

# Local profile (embedded Qdrant)
make docker-up-local
# or
docker compose --profile local up -d --build
```

> Tip: set `PRELOAD_MODELS=1` in `.env` before building if you want the Docker image to pre-download the `BAAI/bge-reranker-base` model.

5. **Open in browser**

```
http://localhost:8501
```

Use `make docker-logs` (or `docker compose --profile cloud logs -f`) to tail the container logs. Run `make docker-down` to stop the active profile, or `DOCKER_PROFILE=local make docker-down` when the embedded Qdrant profile is running.

### Method 2: Local Development

1. **Install dependencies**

```bash
pip install -r requirements.txt
pip install -e .
```

2. **Configure environment variables** (same as above)

3. **Launch the app**

```bash
streamlit run streamlit_app.py
```

## 📋 Features

### 🤖 **Intelligent Q\&A**

* ✅ Professional CDC report analysis
* ✅ Multi‑turn context understanding
* ✅ Real‑time streaming responses
* ✅ Smart question recommendations

### 🔍 **Advanced Retrieval**

* ✅ Hybrid retrieval (Dense + Sparse)
* ✅ Cross‑Encoder re‑ranking
* ✅ Parallel query processing
* ✅ Relevance scoring

### 📊 **User Experience**

* ✅ Modern responsive UI
* ✅ Conversation history
* ✅ Source document display
* ✅ Sample question prompts

### 🛡️ **Enterprise‑Grade**

* ✅ Session persistence
* ✅ Graceful error handling
* ✅ Logging & tracing
* ✅ Health checks & monitoring

## 💻 Usage Examples

### Example questions

```
📋 What did the CDC report about influenza activity among children?
📋 What did the Tennessee report say about early influenza activity in children?
📋 Arthritis—how much higher in young male veterans?
📋 Why do the authors prioritise the WHO STOP AIDS package for children under 5 on ART?
```

### API interface

```python
# Retrieval chain usage
from src.retrieval_graph.graph import build_chain
from src.index_graph.graph import build_index

# Init
index = build_index()
chain = build_chain()

# Query
result = chain.invoke({
    'query': 'your question here',
    **index
})

print(result['answer'])
print(result['sources'])
```

## 🔧 Configuration

### Environment variables

```bash
# Required
OPENAI_API_KEY=your-api-key
QDRANT_URL=your-qdrant-url
QDRANT_API_KEY=your-qdrant-key

# Model config
CHAT_MODEL=gpt-4.1-nano             # Or your vLLM model name
EMBEDDING_MODEL=text-embedding-ada-002

# LLM backend selection
# When set, requests go to your Nginx gateway (OpenAI-compatible) which proxies to vLLM.
# When empty or unset, requests go to OpenAI cloud.
OPENAI_BASE_URL=

# Optional
LANGSMITH_PROJECT=your-project
LANGCHAIN_TRACING_V2=1
```

### Retrieval parameter tuning

```python
# src/retrieval_graph/graph.py
TOP_K = 15      # initial retrieval count
FINAL_K = 5     # final returned count
```

## 📈 Performance Optimisation

### Retrieval performance

* **Parallel queries**: Multiple queries executed concurrently to minimise latency
* **Caching**: Cache vector embeddings
* **Connection pool**: Reuse database connections

### System performance

* **Asynchronous processing**: Non‑blocking I/O
* **Memory optimisation**: Stream large documents
* **Container optimisation**: Multi‑stage Docker builds

## 🚨 Troubleshooting

### Common issues

**1. OpenAI API error**

```bash
# Check API key
echo $OPENAI_API_KEY
# Verify model access
curl -H 'Authorization: Bearer $OPENAI_API_KEY' https://api.openai.com/v1/models
```

**2. Qdrant connection failure**

```bash
# Check network
curl -X GET $QDRANT_URL/collections -H 'api-key: $QDRANT_API_KEY'
```

**3. Docker build failure**

```bash
# Clean and rebuild
docker-compose down --volumes
docker-compose build --no-cache
docker-compose up
```

### Viewing logs

```bash
# Docker logs
docker-compose logs -f rag-app

# App logs
tail -f logs/chat_history_$(date +%Y-%m-%d).jsonl
```

## 🤝 Contribution Guide

### Set up dev environment

```bash
# Install dev deps
pip install -e '.[dev]'

# Code formatting
make format

# Run tests
make test

# Lint
make lint
```



## 🙏 Acknowledgments

* [LangChain](https://github.com/langchain-ai/langchain) – composable LLM application framework
* [Streamlit](https://streamlit.io/) – elegant web app framework
* [Qdrant](https://qdrant.tech/) – high‑performance vector DB
* [OpenAI](https://openai.com/) – advanced AI models

---

**⭐ If this project is helpful, please give it a star!**
