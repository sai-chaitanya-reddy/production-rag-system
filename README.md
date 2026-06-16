# Production-Grade Reliable RAG System

An advanced, enterprise-standard Retrieval-Augmented Generation (RAG) assistant designed for high-precision document intelligence with built-in factual grounding and protection guardrails.

## 🚀 Architectural Blueprint

- **Ingestion & Processing Matrix:** Built-in support for mixed document layouts (PDF, HTML, CSV) with recursive character text segmentation and overlap management.
- **Hybrid Search Engine:** Combines Dense Vector embeddings via ChromaDB and Sparse keyword matching (BM25) to prevent retrieval gaps.
- **Neural Re-ranking Tier:** Cross-Encoder models (`ms-marco-MiniLM-L-6-v2`) grade context relevance dynamically before payload assembly.
- **Factual Guardrails:** Strict prompt boundaries coupled with real-time text validation layers prevent system hallucinations and handle adversarial prompt injections.

## 🛠️ Local Installation & Setup

1. **Clone and Navigate:**
   ```bash
   cd rag_system
   ```
