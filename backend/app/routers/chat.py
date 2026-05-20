from fastapi import APIRouter
import httpx
from app.config import settings
from app.ingestion import search_documents
from app.security import mask_pii, filter_results_by_role
from app.evaluation import get_cached_response, cache_response, log_query

router = APIRouter()

# In-memory session store
_sessions = {}

def calculate_confidence(filtered_results: list) -> dict:
    """Calculate confidence score based on search relevance scores."""
    if not filtered_results:
        return {"score": 0.0, "label": "No Data", "color": "gray"}

    avg_score = sum(r["score"] for r in filtered_results) / len(filtered_results)
    top_score = max(r["score"] for r in filtered_results)
    final_score = (avg_score * 0.4) + (top_score * 0.6)
    percentage = round(final_score * 100, 1)

    if percentage >= 70:
        label = "High"
        color = "green"
    elif percentage >= 45:
        label = "Medium"
        color = "orange"
    else:
        label = "Low"
        color = "red"

    return {"score": percentage, "label": label, "color": color}

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

    # Step 6: Calculate confidence
    confidence = calculate_confidence(filtered_results)

    # Step 7: Build context from filtered results
    if filtered_results:
        context = "\n\n".join([r["text"] for r in filtered_results])
        sources = list(set([r["source"] for r in filtered_results]))
    else:
        context = "No relevant documents found."
        sources = []

    # Step 8: Build conversation history string
    history_text = ""
    if history:
        history_text = "\n".join([
            f"User: {h['question']}\nAssistant: {h['answer']}"
            for h in history[-4:]
        ])
        history_text = f"\nConversation so far:\n{history_text}\n"

    # Step 9: Build RAG prompt with memory
    prompt = f"""You are an enterprise assistant. Answer the user's question using ONLY the context provided below.
If the answer is not in the context, say "I don't have that information in my knowledge base."
{history_text}
Context:
{context}

Question: {clean_question}

Answer:"""

    # Step 10: Send to Llama 3
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

    # Step 11: Save to session memory
    history.append({"question": clean_question, "answer": answer})
    _sessions[session_id] = history[-10:]

    result = {
        "response": answer,
        "sources": sources,
        "role": role,
        "cached": False,
        "session_id": session_id,
        "history_length": len(_sessions[session_id]),
        "confidence": confidence
    }

    # Step 12: Log and cache
    log_query(question=question, sources=sources, role=role, cached=False)
    cache_response(question, result)

    return result

@router.delete("/chat/session/{session_id}")
async def clear_session(session_id: str):
    if session_id in _sessions:
        del _sessions[session_id]
    return {"status": "cleared", "session_id": session_id}
