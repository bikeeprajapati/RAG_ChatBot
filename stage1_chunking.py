# stage1_chunking.py

import fitz  # this is PyMuPDF — "fitz" is its old name, don't let it confuse you

# ── STEP 1: Extract raw text from the PDF ──────────────────────────────
def load_pdf(path):
    doc = fitz.open(path)          # opens the PDF file
    full_text = ""
    for page in doc:
        full_text += page.get_text()   # extract plain text from each page
    return full_text

# ── STEP 2: Split into chunks ──────────────────────────────────────────
def chunk_text(text, chunk_size=500, overlap=50):
    """
    chunk_size = how many characters per chunk
    overlap    = how many characters to repeat between chunks
    so context isn't lost at boundaries
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap  # move forward, but repeat 'overlap' chars

    return chunks


# ── STEP 3: Run it and inspect ─────────────────────────────────────────
if __name__ == "__main__":
    # 👇 replace this with your actual PDF path
    raw_text = load_pdf("data/Bikee_Prajapati.pdf")

    print(f"Total characters extracted: {len(raw_text)}")
    print(f"First 300 chars:\n{raw_text[:300]}")
    print("\n" + "─"*50 + "\n")

    chunks = chunk_text(raw_text, chunk_size=500, overlap=50)

    print(f"Total chunks created: {len(chunks)}")
    print(f"\nChunk #1:\n{chunks[0]}")
    print(f"\nChunk #2 (notice overlap with chunk 1):\n{chunks[1]}")