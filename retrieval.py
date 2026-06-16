import numpy as np
from typing import List, Tuple, Dict, Any

import chromadb
from config import config
from ingestion import get_embedder, collection  # ✅ reuse same model + collection, no duplicate load

def retrieve(query: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Retrieve top-k relevant documents from ChromaDB using vector similarity."""
    embedder = get_embedder()  # ✅ returns already-loaded model, no re-loading
    query_embedding = embedder.encode(query).tolist()

    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=config.TOP_K_RETRIEVAL,
            include=["documents", "metadatas", "distances"]
        )
    except Exception as e:
        print(f"Retrieval error: {e}")
        return [], []

    docs = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    if not docs:
        return [], []

    context_docs = []
    sources = []
    for doc, meta in zip(docs, metadatas):
        context_docs.append({"content": doc, "metadata": meta})
        sources.append(meta.get("source", "unknown"))

    return context_docs, sources
