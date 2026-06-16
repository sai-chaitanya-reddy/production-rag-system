import os
import re
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import config

# Load embedding model once
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Setup ChromaDB
chroma_client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
collection = chroma_client.get_or_create_collection(
    name=config.COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)

# Text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.CHUNK_SIZE,
    chunk_overlap=config.CHUNK_OVERLAP,
    separators=["\n\n", "\n", ".", " "]
)

def load_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def load_html(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    return soup.get_text(separator="\n")

def load_csv(file_path: str) -> str:
    df = pd.read_csv(file_path)
    return df.to_string(index=False)

def load_document(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return load_pdf(file_path)
    elif ext in [".html", ".htm"]:
        return load_html(file_path)
    elif ext == ".csv":
        return load_csv(file_path)
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text.strip()

def ingest_documents(file_paths: List[str]) -> Dict[str, Any]:
    total_chunks = 0
    processed_files = []
    failed_files = []

    for file_path in file_paths:
        try:
            print(f"Processing: {file_path}")
            raw_text = load_document(file_path)
            cleaned = clean_text(raw_text)
            chunks = splitter.split_text(cleaned)

            if not chunks:
                failed_files.append(file_path)
                continue

            embeddings = embedder.encode(chunks).tolist()
            file_name = Path(file_path).name

            ids = [f"{file_name}_chunk_{i}" for i in range(len(chunks))]
            metadatas = [
                {
                    "source": file_name,
                    "file_path": file_path,
                    "chunk_index": i,
                    "file_type": Path(file_path).suffix.lower()
                }
                for i in range(len(chunks))
            ]

            # Add to ChromaDB in batches of 100
            batch_size = 100
            for i in range(0, len(chunks), batch_size):
                collection.upsert(
                    ids=ids[i:i+batch_size],
                    embeddings=embeddings[i:i+batch_size],
                    documents=chunks[i:i+batch_size],
                    metadatas=metadatas[i:i+batch_size]
                )

            total_chunks += len(chunks)
            processed_files.append(file_name)
            print(f"  ✓ {file_name}: {len(chunks)} chunks")

        except Exception as e:
            print(f"  ✗ Failed {file_path}: {e}")
            failed_files.append(file_path)

    return {
        "processed": processed_files,
        "failed": failed_files,
        "total_chunks": total_chunks
    }

def get_collection_stats() -> Dict[str, Any]:
    count = collection.count()
    return {
        "total_documents": count,
        "collection_name": config.COLLECTION_NAME,
        "embedding_model": "all-MiniLM-L6-v2"
    }