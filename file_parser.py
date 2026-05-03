from pathlib import Path
from pypdf import PdfReader
from docx import Document


def extract_text(file_path: str) -> str:
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")

    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if suffix == ".docx":
        doc = Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs)

    if suffix == ".doc":
        # Legacy .doc is binary. For a demo, try best-effort extraction.
        # For production, use LibreOffice conversion or antiword.
        data = path.read_bytes()
        return data.decode("utf-8", errors="ignore")

    raise ValueError(f"Unsupported file type: {suffix}")
