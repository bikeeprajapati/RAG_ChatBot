import fitz
import os

def load_pdf(path: str) -> str:
    """Extract raw text from a PDF file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"PDF not found at: {path}")
    doc = fitz.open(path)
    return "".join(page.get_text() for page in doc)

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split text into overlapping chunks.
    chunk_size: characters per chunk
    overlap: characters shared between consecutive chunks
    """
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks