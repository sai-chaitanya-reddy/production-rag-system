import os
import shutil
from pathlib import Path
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from config import config
from ingestion import ingest_documents, get_collection_stats
from retrieval import retrieve
from generation import generate_response_stream
from reliability import check_prompt_injection

app = FastAPI(
    title="Production-Grade Engineered RAG System",
    description="FastAPI + ChromaDB + Local Hybrid/Re-ranking + Groq Engine",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("./uploaded_files")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.get("/")
def read_root():
    return {
        "status": "healthy",
        "system": "Production-Grade RAG API Pipeline Active"
    }

@app.get("/stats")
def read_stats():
    try:
        return get_collection_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_documents(file: UploadFile = File(..., description="Select your PDF, CSV, or HTML file")):
    """Accepts a PDF, CSV, or HTML file, stores it locally, and builds Vector Embeddings."""
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        results = ingest_documents([str(file_path)])
        return {
            "message": "File processed and indexed successfully",
            "ingestion_metrics": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion lifecycle aborted: {str(e)}")

@app.get("/chat")
async def chat_stream(
    query: str = Query(..., description="User prompt query"),
    session_id: str = Query("default_session", description="Unique conversation session tracking label")
):
    if check_prompt_injection(query):
        async def malicious_alert():
            yield "[Security Alert]: Malicious instruction pattern detected. Query processing rejected."
        return StreamingResponse(malicious_alert(), media_type="text/plain")

    try:
        context_docs, sources = retrieve(query)
        if not context_docs:
            async def empty_context_fallback():
                yield "No relevant context could be fetched from the reference library. Please upload target documentation files."
            return StreamingResponse(empty_context_fallback(), media_type="text/plain")
            
        return StreamingResponse(
            generate_response_stream(query, context_docs, sources, session_id),
            media_type="text/event-stream"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Streaming core pipeline failure: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)