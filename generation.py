from typing import AsyncGenerator, List, Dict, Any
from groq import AsyncGroq
from config import config

# Initialize Async Groq Client
client = AsyncGroq(api_key=config.GROQ_API_KEY)

# Simple in-memory session store
SESSION_MEMORY: Dict[str, List[Dict[str, str]]] = {}

def get_session_history(session_id: str) -> List[Dict[str, str]]:
    if session_id not in SESSION_MEMORY:
        SESSION_MEMORY[session_id] = []
    return SESSION_MEMORY[session_id][-config.MAX_HISTORY:]

def add_to_session_history(session_id: str, role: str, content: str):
    if session_id not in SESSION_MEMORY:
        SESSION_MEMORY[session_id] = []
    SESSION_MEMORY[session_id].append({"role": role, "content": content})

async def generate_response_stream(
    query: str,
    context_docs: List[Dict[str, Any]],
    sources: List[str],
    session_id: str
) -> AsyncGenerator[str, None]:
    """Generates streaming responses using Groq."""

    context_str = ""
    for i, doc in enumerate(context_docs):
        source_name = doc["metadata"].get("source", "unknown")
        context_str += f"[Doc {i+1}] (Source: {source_name}):\n{doc['content']}\n\n"

    system_prompt = (
        "You are a helpful AI assistant inside a RAG system.\n"
        "Answer the user query using ONLY the provided contexts below.\n\n"
        f"--- CONTEXT START ---\n{context_str or 'No context available.'}\n--- CONTEXT END ---\n\n"
        "RULES:\n"
        "1. Use only facts from the context.\n"
        "2. If context is insufficient, say you don't know.\n"
        "3. Cite sources like [Doc 1], [Doc 2].\n"
        "4. Be concise and clear."
    )

    history = get_session_history(session_id)
    messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": query}]

    try:
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

        add_to_session_history(session_id, "user", query)
        add_to_session_history(session_id, "assistant", full_response)

    except Exception as e:
        yield f"\n[Error]: {str(e)}"
