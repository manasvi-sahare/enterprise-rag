from fastapi import APIRouter
from app.evaluation import log_feedback, get_feedback_stats, get_analytics

router = APIRouter()

@router.post("/feedback")
async def submit_feedback(payload: dict):
    question = payload.get("question", "")
    context = payload.get("context", "")
    answer = payload.get("answer", "")
    sources = payload.get("sources", [])
    role = payload.get("role", "guest")
    feedback = payload.get("feedback", "positive")

    if feedback not in ["positive", "negative"]:
        return {"error": "feedback must be 'positive' or 'negative'"}

    result = log_feedback(
        question=question,
        context=context,
        answer=answer,
        sources=sources,
        role=role,
        feedback=feedback
    )
    return result

@router.get("/feedback/stats")
async def feedback_stats():
    return get_feedback_stats()

@router.get("/analytics")
async def analytics():
    return get_analytics()
