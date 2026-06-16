import os
import requests
from sentence_transformers import SentenceTransformer
import numpy as np

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

HF_API_URL = "https://api-inference.huggingface.co/models/cross-encoder/ms-marco-MiniLM-L-6-v2"
HF_TOKEN = os.getenv("HF_TOKEN", "")
headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

def hybrid_retrieval(query: str, documents: list, top_k: int = 3):
    if not documents:
        return []
    
    query_vector = embedding_model.encode(query)
    doc_vectors = embedding_model.encode(documents)
    
    scores = np.dot(doc_vectors, query_vector) / (np.linalg.norm(doc_vectors, axis=1) * np.linalg.norm(query_vector) + 1e-9)
    top_indices = np.argsort(scores)[::-1][:top_k * 2]
    candidate_docs = [documents[i] for i in top_indices]
    
    try:
        payload = {
            "inputs": {
                "source_sentence": query,
                "sentences": candidate_docs
            }
        }
        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=5)
        if response.status_code == 200:
            api_scores = response.json()
            if isinstance(api_scores, list) and len(api_scores) == len(candidate_docs):
                ranked_pairs = sorted(zip(candidate_docs, api_scores), key=lambda x: x[1], reverse=True)
                return [doc for doc, score in ranked_pairs[:top_k]]
    except Exception:
        pass
        
    return candidate_docs[:top_k]
