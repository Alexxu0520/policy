from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from file_parser import extract_text
from rag_models import audit_document

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Policy RAG Audit System - Qwen")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


@app.get("/")
def home():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.post("/api/audit")
async def audit_file(file: UploadFile = File(...)):
    allowed = {".txt", ".pdf", ".doc", ".docx"}
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail="Only .txt, .pdf, .doc, .docx are supported")

    save_path = UPLOAD_DIR / f"{uuid4().hex}{suffix}"
    content = await file.read()
    save_path.write_bytes(content)

    try:
        text = extract_text(str(save_path)).strip()
        if not text:
            raise ValueError("No text extracted from file")
        result = audit_document(text)
        return {
            "filename": file.filename,
            "extracted_preview": text[:1000],
            **result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/audit-text")
async def audit_text(payload: dict):
    text = (payload.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    return audit_document(text)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
