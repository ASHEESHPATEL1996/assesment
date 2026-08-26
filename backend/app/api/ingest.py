import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.rag.ingest import ingest_file, list_documents

router = APIRouter()
ALLOWED = {".pdf", ".docx", ".md", ".txt"}


@router.get("/documents")
def get_documents() -> dict:
    return {"documents": list_documents()}


@router.post("/documents")
def upload_document(file: UploadFile = File(...)) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name
    try:
        return ingest_file(tmp_path, file.filename or Path(tmp_path).name)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        Path(tmp_path).unlink(missing_ok=True)
