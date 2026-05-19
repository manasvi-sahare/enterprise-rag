from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import uuid
import os

# Initialize embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize Qdrant in-memory
client = QdrantClient(":memory:")

COLLECTION_NAME = "enterprise_docs"

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)

# BM25 index stored in memory
_all_chunks = []  # list of {"text": ..., "source": ..., "clearance_level": ..., "department": ..., "author": ...}
_bm25_index = None

def _rebuild_bm25():
    global _bm25_index
    if _all_chunks:
        tokenized = [chunk["text"].lower().split() for chunk in _all_chunks]
        _bm25_index = BM25Okapi(tokenized)

def parse_document(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())

    elif ext == ".docx":
        from docx import Document
        doc = Document(file_path)
        return "\n".join(para.text for para in doc.paragraphs if para.text.strip())

    elif ext == ".txt":
        for encoding in ["utf-8", "utf-16", "latin-1"]:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        raise ValueError("Could not decode text file")

    else:
        raise ValueError(f"Unsupported file type: {ext}")

def chunk_text(text: str, chunk_size: int = 500) -> list:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk) > 50:
            chunks.append(chunk)
    return chunks

def ingest_document(file_path: str, metadata: dict):
    global _all_chunks
    text = parse_document(file_path)
    chunks = chunk_text(text)

    points = []
    for chunk in chunks:
        embedding = embedder.encode(chunk).tolist()
        chunk_data = {
            "text": chunk,
            "source": os.path.basename(file_path),
            **metadata
        }
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload=chunk_data
        )
        points.append(point)
        _all_chunks.append(chunk_data)

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    _rebuild_bm25()
    return {"chunks_stored": len(points)}

def search_documents(query: str, top_k: int = 5):
    # Dense vector search
    query_vector = embedder.encode(query).tolist()
    vector_results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k
    ).points

    vector_hits = {
        r.payload["text"]: {
            "text": r.payload["text"],
            "source": r.payload["source"],
            "score": r.score,
            "clearance_level": r.payload.get("clearance_level", "public"),
            "department": r.payload.get("department", "general"),
            "author": r.payload.get("author", "unknown")
        }
        for r in vector_results
    }

    # BM25 keyword search
    bm25_hits = {}
    if _bm25_index and _all_chunks:
        tokenized_query = query.lower().split()
        bm25_scores = _bm25_index.get_scores(tokenized_query)
        top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]
        for idx in top_indices:
            if bm25_scores[idx] > 0:
                chunk = _all_chunks[idx]
                bm25_hits[chunk["text"]] = {
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "score": float(bm25_scores[idx]) / 10,
                    "clearance_level": chunk.get("clearance_level", "public"),
                    "department": chunk.get("department", "general"),
                    "author": chunk.get("author", "unknown")
                }

    # Merge results - combine scores for overlapping hits
    merged = {}
    for text, hit in vector_hits.items():
        merged[text] = hit.copy()
        if text in bm25_hits:
            merged[text]["score"] = (hit["score"] + bm25_hits[text]["score"]) / 2

    for text, hit in bm25_hits.items():
        if text not in merged:
            merged[text] = hit

    # Sort by final score
    final_results = sorted(merged.values(), key=lambda x: x["score"], reverse=True)[:top_k]
    return final_results
