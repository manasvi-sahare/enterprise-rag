import json
import os
from datetime import datetime

FEEDBACK_LOG = "data/feedback_log.jsonl"
QUERY_LOG = "data/query_log.jsonl"

def log_query(question: str, sources: list, role: str, cached: bool):
    """Log every query for analytics."""
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "question": question,
        "sources": sources,
        "role": role,
        "cached": cached
    }
    os.makedirs("data", exist_ok=True)
    with open(QUERY_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

def log_feedback(question: str, context: str, answer: str, sources: list, role: str, feedback: str):
    """Log user feedback for future fine-tuning and evaluation."""
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "question": question,
        "context": context,
        "answer": answer,
        "sources": sources,
        "role": role,
        "feedback": feedback
    }
    os.makedirs("data", exist_ok=True)
    with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return {"status": "logged", "feedback": feedback}

def get_analytics():
    """Return full analytics stats."""
    # Query stats
    total_queries = 0
    cached_queries = 0
    role_counts = {}
    doc_counts = {}

    if os.path.exists(QUERY_LOG):
        with open(QUERY_LOG, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line.strip())
                total_queries += 1
                if entry.get("cached"):
                    cached_queries += 1
                role = entry.get("role", "unknown")
                role_counts[role] = role_counts.get(role, 0) + 1
                for src in entry.get("sources", []):
                    doc_counts[src] = doc_counts.get(src, 0) + 1

    # Feedback stats
    total_feedback = 0
    positive = 0
    negative = 0

    if os.path.exists(FEEDBACK_LOG):
        with open(FEEDBACK_LOG, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line.strip())
                total_feedback += 1
                if entry["feedback"] == "positive":
                    positive += 1
                else:
                    negative += 1

    return {
        "queries": {
            "total": total_queries,
            "cached": cached_queries,
            "cache_hit_rate": round(cached_queries / total_queries * 100, 1) if total_queries > 0 else 0,
            "by_role": role_counts,
            "top_documents": sorted(doc_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        },
        "feedback": {
            "total": total_feedback,
            "positive": positive,
            "negative": negative,
            "positive_rate": round(positive / total_feedback * 100, 1) if total_feedback > 0 else 0
        }
    }

def get_feedback_stats():
    analytics = get_analytics()
    return analytics["feedback"]

# Simple semantic cache
_cache = {}

def get_cached_response(question: str):
    return _cache.get(question.strip().lower())

def cache_response(question: str, response: dict):
    _cache[question.strip().lower()] = response
