from fastapi import APIRouter, UploadFile, File, Form
from app.ingestion import ingest_document, search_documents, get_all_documents, delete_document
from app.security import filter_results_by_role, mask_pii
import shutil
import os

router = APIRouter()

UPLOAD_DIR = "data/sample_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    department: str = Form(default="general"),
    author: str = Form(default="unknown"),
    clearance_level: str = Form(default="public")
):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = ingest_document(
        file_path=file_path,
        metadata={
            "department": department,
            "author": author,
            "clearance_level": clearance_level
        }
    )

    return {
        "filename": file.filename,
        "chunks_stored": result["chunks_stored"],
        "status": "success"
    }

@router.get("/search")
async def search(query: str, top_k: int = 5, role: str = "guest"):
    clean_query = mask_pii(query)
    results = search_documents(query=clean_query, top_k=top_k)
    filtered = filter_results_by_role(results, role)
    return {"results": filtered, "role": role}

@router.get("/list")
async def list_documents():
    return {"documents": get_all_documents()}

@router.delete("/delete/{filename}")
async def delete_doc(filename: str):
    result = delete_document(filename)
    file_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
    return result
