import os
import requests

HF_TOKEN = os.getenv("HF_TOKEN", "")
headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

EMBED_API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
RERANK_API_URL = "https://api-inference.huggingface.co/models/cross-encoder/ms-marco-MiniLM-L-6-v2"

def get_hf_embeddings(texts):
    try:
        response = requests.post(EMBED_API_URL, headers=headers, json={"inputs": texts, "options": {"wait_for_model": True}}, timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None

def hybrid_retrieval(query: str, documents: list, top_k: int = 3):
    if not documents:
        return []
    
    import numpy as np
    
    all_texts = [query] + documents
    embeddings = get_hf_embeddings(all_texts)
    
    if embeddings and len(embeddings) == len(all_texts):
        query_vector = np.array(embeddings[0])
        doc_vectors = np.array(embeddings[1:])
        
        scores = np.dot(doc_vectors, query_vector) / (np.linalg.norm(doc_vectors, axis=1) * np.linalg.norm(query_vector) + 1e-9)
        top_indices = np.argsort(scores)[::-1][:top_k * 2]
        candidate_docs = [documents[i] for i in top_indices]
    else:
        candidate_docs = documents[:top_k * 2]
        
    try:
        payload = {
            "inputs": {
                "source_sentence": query,
                "sentences": candidate_docs
            }
        }
        response = requests.post(RERANK_API_URL, headers=headers, json=payload, timeout=5)
        if response.status_code == 200:
            api_scores = response.json()
            if isinstance(api_scores, list) and len(api_scores) == len(candidate_docs):
                ranked_pairs = sorted(zip(candidate_docs, api_scores), key=lambda x: x[1], reverse=True)
                return [doc for doc, score in ranked_pairs[:top_k]]
    except Exception:
        pass
        
    return candidate_docs[:top_k]
