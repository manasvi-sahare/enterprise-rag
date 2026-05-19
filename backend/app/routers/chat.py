from fastapi import APIRouter
import httpx
from app.config import settings
from app.ingestion import search_documents
from app.security import mask_pii, filter_results_by_role
from app.evaluation import get_cached_response, cache_response, log_query

router = APIRouter()

@router.post("/chat")
async def chat(payload: dict):
    question = payload.get("message", "")
    role = payload.get("role", "guest")

    # Step 1: Check semantic cache first
    cached = get_cached_response(question)
    if cached:
        log_query(question=question, sources=cached.get("sources", []), role=role, cached=True)
        return {**cached, "cached": True}

    # Step 2: Mask PII in the incoming question
    clean_question = mask_pii(question)

    # Step 3: Search for relevant documents
    search_results = search_documents(query=clean_question, top_k=3)

    # Step 4: Filter results based on user role
    filtered_results = filter_results_by_role(search_results, role)

    # Step 5: Build context from filtered results
    if filtered_results:
        context = "\n\n".join([r["text"] for r in filtered_results])
        sources = list(set([r["source"] for r in filtered_results]))
    else:
        context = "No relevant documents found."
        sources = []

    # Step 6: Build RAG prompt
    prompt = f"""You are an enterprise assistant. Answer the user's question using ONLY the context provided below.
If the answer is not in the context, say "I don't have that information in my knowledge base."

Context:
{context}

Question: {clean_question}

Answer:"""

    # Step 7: Send to Llama 3
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
    result = {
        "response": data.get("response", ""),
        "sources": sources,
        "role": role,
        "cached": False
    }

    # Step 8: Log query and cache response
    log_query(question=question, sources=sources, role=role, cached=False)
    cache_response(question, result)

    return result
