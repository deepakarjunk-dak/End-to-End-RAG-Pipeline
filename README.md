# End-to-End RAG Pipeline: From Document Ingestion to Agentic Retrieval

A progressive, hands-on implementation of Retrieval Augmented Generation (RAG) — built across four modules that evolve from foundational document pipelines to a fully agentic, graph-based retrieval system using LangGraph.

---

## Overview

This repository explores RAG at increasing levels of sophistication. Each module introduces a new concept, tool, or architectural pattern — collectively covering the full lifecycle of a production-grade RAG system.

| Module | Focus | Key Tech |
|--------|-------|----------|
| `01_document_pipeline` | Document ingestion, chunking, vector storage, simple + advanced RAG | ChromaDB, SentenceTransformers, Groq |
| `02_mongodb_rag` | Cloud-native vector storage with MongoDB Atlas | MongoDB Atlas Vector Search, Groq |
| `03_typesense_rag` | Hybrid keyword + semantic search | Typesense, HuggingFace Embeddings, Groq |
| `04_agentic_rag_langgraph` | Stateful agentic pipeline with conditional retrieval logic | LangGraph, FAISS, Groq |

---

## Architecture

### Core RAG Flow (Modules 1–3)

```
  ┌─────────────────────────────────────────────────────────────┐
  │                     DATA INGESTION                          │
  │  PDF / TXT / Directory  →  Loader  →  Text Splitter         │
  │         (LangChain Loaders)    (RecursiveCharacterSplitter) │
  └───────────────────────┬─────────────────────────────────────┘
                          │  chunks
                          ▼
  ┌─────────────────────────────────────────────────────────┐
  │                     EMBEDDING                           │
  │     SentenceTransformer / HuggingFaceEmbeddings         │
  │          (all-MiniLM-L6-v2  →  384-dim vectors)         │
  └───────────────────────┬─────────────────────────────────┘
                          │  vectors
                          ▼
  ┌─────────────────────────────────────────────────────────┐
  │                   VECTOR STORE                          │
  │   ChromaDB  ──or──  MongoDB Atlas  ──or──  Typesense    │
  │   (local)          (cloud)               (hybrid search)│
  └───────────────────────┬─────────────────────────────────┘
                          │
          ┌───────────────┴───────────────┐
          │         QUERY TIME            │
          ▼                               ▼
    User Query                    Embed Query
          │                               │
          └───────────────┬───────────────┘
                          │
                          ▼
             Cosine Similarity Search
             (Top-K relevant chunks)
                          │
                          ▼
  ┌─────────────────────────────────────────────────────────┐
  │                  LLM GENERATION                         │
  │       Groq API  →  Llama-3.1-8b-instant                 │
  │   Context + Query  →  Grounded Answer                   │
  └─────────────────────────────────────────────────────────┘
```

### Agentic RAG Flow (Module 4 — LangGraph)

```
                    User Question
                         │
                         ▼
              ┌──────────────────────┐
              │    DECIDE NODE       │
              │  (needs retrieval?)  │
              └────────┬─────────────┘
                       │
           ┌───────────┴───────────┐
           │                       │
      YES (retrieve)          NO (direct)
           │                       │
           ▼                       │
  ┌─────────────────┐              │
  │  RETRIEVE NODE  │              │
  │  FAISS Vector   │              │
  │  Similarity     │              │
  │  Search         │              │
  └────────┬────────┘              │
           │                       │
           └───────────┬───────────┘
                       │
                       ▼
            ┌─────────────────────┐
            │   GENERATE NODE     │
            │  Groq LLM           │
            │  (RAG or Direct)    │
            └────────┬────────────┘
                     │
                     ▼
                  Answer
```

---

## Tech Stack

- **LLM**: Groq API — `llama-3.1-8b-instant`
- **Embeddings**: HuggingFace `all-MiniLM-L6-v2` via SentenceTransformers
- **Vector Stores**: ChromaDB · MongoDB Atlas · Typesense · FAISS
- **Orchestration**: LangChain · LangGraph
- **Document Loaders**: PyPDFLoader · PyMuPDFLoader · TextLoader · DirectoryLoader
- **Text Splitting**: RecursiveCharacterTextSplitter

---

## Repository Structure

```
rag-pipeline-evolution/
│
├── README.md
├── requirements.txt
├── .env.example
│
├── 01_document_pipeline/
│   └── document.ipynb        # Ingestion, ChromaDB, simple + advanced RAG
│
├── 02_mongodb_rag/
│   └── rag.ipynb             # MongoDB Atlas Vector Search + Groq
│
├── 03_typesense_rag/
│   └── typesense.ipynb       # Typesense hybrid search + Groq
│
└── 04_agentic_rag_langgraph/
    └── agenticrag.ipynb      # LangGraph stateful agentic pipeline
```

---

## Setup & Usage

### 1. Clone the repository

```bash
git clone https://github.com/your-username/rag-pipeline-evolution.git
cd rag-pipeline-evolution
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
# Fill in your API keys in .env
```

### 4. Run any module

Open the desired notebook in Jupyter or VS Code and run cells sequentially.

---

## Environment Variables

Create a `.env` file based on `.env.example`:

```env
GROQ_API_KEY=your_groq_api_key_here
MONGODB_URI=your_mongodb_connection_string_here
TYPESENSE_API_KEY=your_typesense_api_key_here
```

---

## Module Highlights

### Module 1 — Document Pipeline (ChromaDB)
- Custom `EmbeddingManager` class for batch embedding generation
- Custom `VectorStore` class with full CRUD over ChromaDB collections
- `RAGRetriever` with configurable `top_k` and `score_threshold`
- Simple RAG and Enhanced RAG with source citation and confidence scoring

### Module 2 — MongoDB Atlas RAG
- Cloud-hosted vector store with Atlas Vector Search index
- Cosine similarity over 384-dimensional embeddings
- Real-time PDF ingestion from a remote URL

### Module 3 — Typesense RAG
- Hybrid keyword + semantic retrieval via Typesense
- LangChain-native integration using `Typesense` vectorstore wrapper
- Schema-driven document indexing with facet support

### Module 4 — Agentic RAG with LangGraph ⭐
- Stateful graph with typed `AgentState` (TypedDict)
- Decision node determines retrieval vs. direct generation at runtime
- Conditional edges using `should_retrieve()` routing function
- FAISS-backed retriever with Groq LLM generation node
- Visual graph rendering via Mermaid

---

## Key Concepts Demonstrated

- **RAG fundamentals**: chunking strategy, embedding selection, similarity search
- **Multi-vectorstore fluency**: same pipeline adapted across 4 different stores
- **Agentic design**: stateful graph, conditional branching, node-based execution
- **Production patterns**: environment variable management, modular class design, confidence scoring, source attribution

---

## Requirements

```
langchain
langchain-community
langchain-groq
langchain-huggingface
langchain-text-splitters
langgraph
sentence-transformers
chromadb
pymongo
typesense
faiss-cpu
groq
python-dotenv
pypdf
pymupdf
```

---

## Author

**Deepak Arjun K**  
PGP in Data Science & Generative AI | BE Mechanical Engineering
