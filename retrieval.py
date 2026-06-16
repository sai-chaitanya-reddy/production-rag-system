from typing import List, Dict, Any, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from ingestion import collection, embedder
from config import config

# Local re-ranker (no API needed!)
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def get_query_embedding(query: str) -> List[float]:
    return embedder.encode([query])[0].tolist()

def expand_query(query: str) -> List[str]:
    """Generate multiple query variations for better retrieval"""
    queries = [query]
    # Simple expansion strategies
    if "what is" in query.lower():
        queries.append(query.lower().replace("what is", "define"))
        queries.append(query.lower().replace("what is", "explain"))
    if "how" in query.lower():
        queries.append(query.lower().replace("how", "steps to"))
        queries.append(query.lower().replace("how", "process of"))
    return list(set(queries))

def dense_search(query: str, top_k: int = 10) -> List[Dict]:
    """Vector similarity search"""
    embedding = get_query_embedding(query)
    results = collection.query(
        query_embeddings=[embedding],
        n_results=min(top_k, collection.count() or 1),
        include=["documents", "metadatas", "distances"]
    )

    docs = []
    if results["documents"] and results["documents"][0]:
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            docs.append({
                "content": doc,
                "metadata": meta,
                "score": 1 - dist  # convert distance to similarity
            })
    return docs

def bm25_search(query: str, top_k: int = 10) -> List[Dict]:
    """Keyword-based search using ChromaDB where filter"""
    keywords = [w for w in query.lower().split() if len(w) > 3]
    if not keywords:
        return []

    all_docs = collection.get(include=["documents", "metadatas"])
    if not all_docs["documents"]:
        return []

    scored = []
    for doc, meta in zip(all_docs["documents"], all_docs["metadatas"]):
        doc_lower = doc.lower()
        score = sum(doc_lower.count(kw) for kw in keywords)
        if score > 0:
            scored.append({
                "content": doc,
                "metadata": meta,
                "score": score / len(keywords)
            })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]

def hybrid_search(query: str, top_k: int = 10) -> List[Dict]:
    """Combine dense + BM25 results"""
    dense_results = dense_search(query, top_k)
    bm25_results = bm25_search(query, top_k)

    # Merge by content, keeping highest score
    seen = {}
    for doc in dense_results:
        key = doc["content"][:100]
        seen[key] = doc

    for doc in bm25_results:
        key = doc["content"][:100]
        if key not in seen:
            seen[key] = doc
        else:
            # Combine scores
            seen[key]["score"] = (seen[key]["score"] + doc["score"]) / 2

    merged = list(seen.values())
    merged.sort(key=lambda x: x["score"], reverse=True)
    return merged[:top_k]

def rerank(query: str, docs: List[Dict], top_k: int = 5) -> List[Dict]:
    """Re-rank using cross-encoder for better precision"""
    if not docs:
        return []

    pairs = [[query, doc["content"]] for doc in docs]
    scores = reranker.predict(pairs)

    for doc, score in zip(docs, scores):
        doc["rerank_score"] = float(score)

    reranked = sorted(docs, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]

def compress_context(docs: List[Dict], max_tokens: int = 2000) -> List[Dict]:
    """Trim content to stay within token limits"""
    compressed = []
    total_chars = 0
    max_chars = max_tokens * 4  # rough estimate

    for doc in docs:
        content = doc["content"]
        if total_chars + len(content) > max_chars:
            remaining = max_chars - total_chars
            if remaining > 100:
                doc = doc.copy()
                doc["content"] = content[:remaining] + "..."
                compressed.append(doc)
            break
        compressed.append(doc)
        total_chars += len(content)

    return compressed

def retrieve(query: str) -> Tuple[List[Dict], List[str]]:
    """Full retrieval pipeline"""
    if collection.count() == 0:
        return [], ["No documents ingested yet. Please upload documents first."]

    # Step 1: Query expansion
    queries = expand_query(query)

    # Step 2: Hybrid search across all query variations
    all_results = []
    for q in queries:
        results = hybrid_search(q, top_k=10)
        all_results.extend(results)

    # Deduplicate
    seen = {}
    for doc in all_results:
        key = doc["content"][:100]
        if key not in seen or doc["score"] > seen[key]["score"]:
            seen[key] = doc
    unique_results = list(seen.values())

    # Step 3: Re-rank
    reranked = rerank(query, unique_results, top_k=config.TOP_K_RETRIEVAL)

    # Step 4: Compress context
    compressed = compress_context(reranked)

    # Extract sources
    sources = list(set(
        doc["metadata"].get("source", "unknown")
        for doc in compressed
    ))

    return compressed, sources