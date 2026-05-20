from fastapi import APIRouter
import httpx
from app.config import settings
from app.ingestion import search_documents
from app.security import mask_pii, filter_results_by_role
from app.evaluation import get_cached_response, cache_response, log_query

router = APIRouter()

# In-memory session store: session_id -> list of messages
_sessions = {}

@router.post("/chat")
async def chat(payload: dict):
    question = payload.get("message", "")
    role = payload.get("role", "guest")
    session_id = payload.get("session_id", "default")

    # Step 1: Check semantic cache first
    cached = get_cached_response(question)
    if cached:
        log_query(question=question, sources=cached.get("sources", []), role=role, cached=True)
        return {**cached, "cached": True}

    # Step 2: Mask PII in the incoming question
    clean_question = mask_pii(question)

    # Step 3: Get conversation history for this session
    history = _sessions.get(session_id, [])

    # Step 4: Search for relevant documents
    search_results = search_documents(query=clean_question, top_k=3)

    # Step 5: Filter results based on user role
    filtered_results = filter_results_by_role(search_results, role)

    # Step 6: Build context from filtered results
    if filtered_results:
        context = "\n\n".join([r["text"] for r in filtered_results])
        sources = list(set([r["source"] for r in filtered_results]))
    else:
        context = "No relevant documents found."
        sources = []

    # Step 7: Build conversation history string
    history_text = ""
    if history:
        history_text = "\n".join([
            f"User: {h['question']}\nAssistant: {h['answer']}"
            for h in history[-4:]  # last 4 exchanges
        ])
        history_text = f"\nConversation so far:\n{history_text}\n"

    # Step 8: Build RAG prompt with memory
    prompt = f"""You are an enterprise assistant. Answer the user's question using ONLY the context provided below.
If the answer is not in the context, say "I don't have that information in my knowledge base."
{history_text}
Context:
{context}

Question: {clean_question}

Answer:"""

    # Step 9: Send to Llama 3
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            f"{settings.ollama_base_url}/api/generate",
            json={
                "model": settings.ollama_model,
                "prompt": prompt,
                "stream": False
            }
        )

    data = response.json()
    answer = data.get("response", "")

    # Step 10: Save to session memory
    history.append({"question": clean_question, "answer": answer})
    _sessions[session_id] = history[-10:]  # keep last 10 exchanges

    result = {
        "response": answer,
        "sources": sources,
        "role": role,
        "cached": False,
        "session_id": session_id,
        "history_length": len(_sessions[session_id])
    }

    # Step 11: Log and cache
    log_query(question=question, sources=sources, role=role, cached=False)
    cache_response(question, result)

    return result

@router.delete("/chat/session/{session_id}")
async def clear_session(session_id: str):
    if session_id in _sessions:
        del _sessions[session_id]
    return {"status": "cleared", "session_id": session_id}
