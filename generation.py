import os
from typing import AsyncGenerator, List, Dict, Any
from groq import AsyncGroq
from config import config

# Initialize Async Groq Client
client = AsyncGroq(api_key=config.GROQ_API_KEY)

# Simple in-memory session store for multi-turn history
SESSION_MEMORY: Dict[str, List[Dict[str, str]]] = {}

def get_session_history(session_id: str) -> List[Dict[str, str]]:
    """Retrieve history up to the configured max memory limits."""
    if session_id not in SESSION_MEMORY:
        SESSION_MEMORY[session_id] = []
    return SESSION_MEMORY[session_id][-config.MAX_HISTORY:]

def add_to_session_history(session_id: str, role: str, content: str):
    """Append a new message turn to session memory."""
    if session_id not in SESSION_MEMORY:
        SESSION_MEMORY[session_id] = []
    SESSION_MEMORY[session_id].append({"role": role, "content": content})

async def generate_response_stream(
    query: str, 
    context_docs: List[Dict[str, Any]], 
    sources: List[str], 
    session_id: str
) -> AsyncGenerator[str, None]:
    """Generates streaming responses using Groq with robust context integration and citations."""
    
    # Format retrieved contexts cleanly
    context_str = ""
    for i, doc in enumerate(context_docs):
        source_name = doc["metadata"].get("source", "unknown")
        context_str += f"[Doc {i+1}] (Source: {source_name}):\n{doc['content']}\n\n"
        
    system_prompt = (
        "You are an advanced, production-grade AI technical assistant operating inside a RAG system.\n"
        "Your task is to answer the user query accurately and comprehensively using ONLY the provided contexts below.\n\n"
        f"--- CONTEXT START ---\n{context_str or 'No context available.'}\n--- CONTEXT END ---\n\n"
        "RULES FOR GENERATION:\n"
        "1. Prioritize facts explicitly stated in the context.\n"
        "2. If the context does not contain enough information to answer, state clearly that you do not know based on the provided material. Do not hallucinate or guess.\n"
        "3. Inline citations: When mentioning information from a document, use inline numeric markers like [Doc 1], [Doc 2].\n"
        "4. Remain concise, clear, and professional."
    )
    
    # Compile messages including conversational history
    history = get_session_history(session_id)
    messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": query}]
    
    try:
        # Request stream from Groq
        response_stream = await client.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=1024,
            stream=True
        )
        
        full_response = ""
        async for chunk in response_stream:
            content = chunk.choices[0].delta.content or ""
            if content:
                full_response += content
                yield content
                
        # Save generated response back to conversational memory
        add_to_session_history(session_id, "user", query)
        add_to_session_history(session_id, "assistant", full_response)
        
    except Exception as e:
        yield f"\n[Generation Error]: Failed to stream response from Groq engine: {str(e)}"