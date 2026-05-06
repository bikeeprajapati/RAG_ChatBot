# main.py
# ─────────────────────────────────────────────────────────────────────
# RESPONSIBILITY: HTTP layer — expose the RAG pipeline as a web API.
#
# Endpoints:
#   GET  /         → serves frontend HTML
#   POST /upload   → accepts PDF, processes it, builds vector store
#   POST /ask      → accepts question + history, returns answer
#   GET  /health   → confirms server is alive
# ─────────────────────────────────────────────────────────────────────

import sys
import os
import logging

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

sys.path.append("src")

from chunking     import load_pdf, chunk_text
from embedder     import embed_chunks
from vector_store import build_vector_store, load_vector_store, search
from chatbot      import generate_answer

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
logger = logging.getLogger(__name__)


# ── APP SETUP ─────────────────────────────────────────────────────────
app = FastAPI(
    title="DocMind — PDF RAG Chatbot",
    description="Upload any PDF and chat with it",
    version="1.0.0"
)

app.mount("/static", StaticFiles(directory="frontend"), name="static")


# ── CONSTANTS ─────────────────────────────────────────────────────────
UPLOAD_DIR = "data"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ── APP STATE ─────────────────────────────────────────────────────────
# Holds the current session's vector index and chunks in memory
# Simple dict — works fine for single-user app
# Multi-user production: use Redis or a database instead
rag_state = {
    "index":  None,
    "chunks": None,
    "ready":  False
}


# ── REQUEST / RESPONSE MODELS ─────────────────────────────────────────
class QuestionRequest(BaseModel):
    question: str
    history:  list[dict] = []
    # history is optional — defaults to empty list
    # Each item: {"role": "user" | "assistant", "content": "..."}
    # Pydantic validates this automatically — no manual parsing needed

class AnswerResponse(BaseModel):
    question: str
    answer:   str


# ── ROUTES ────────────────────────────────────────────────────────────
@app.get("/")
async def serve_frontend():
    """Serves the chat UI to the browser."""
    return FileResponse("frontend/index.html")


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Accepts a PDF upload and processes it through the full RAG pipeline.

    Flow:
    1. Validate file type
    2. Save to disk
    3. Chunk text
    4. Embed chunks via HF API
    5. Build FAISS index
    6. Load into memory
    7. Set ready = True
    """

    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted."
        )

    # Save uploaded file to disk
    pdf_path = os.path.join(UPLOAD_DIR, "uploaded.pdf")

    with open(pdf_path, "wb") as f:
        content = await file.read()
        # await = async read — doesn't block other requests while reading
        f.write(content)

    logger.info(f"PDF saved → {pdf_path} ({len(content)} bytes)")

    try:
        # Step 1: Chunk
        logger.info("Chunking PDF...")
        text   = load_pdf(pdf_path)
        chunks = chunk_text(text)
        logger.info(f"Created {len(chunks)} chunks")

        # Step 2: Embed
        logger.info("Embedding chunks...")
        embeddings = embed_chunks(chunks)

        # Step 3: Build and save FAISS index
        logger.info("Building vector store...")
        build_vector_store(chunks, embeddings, save_dir=UPLOAD_DIR)

        # Step 4: Load into memory
        index, loaded_chunks = load_vector_store(save_dir=UPLOAD_DIR)

        # Step 5: Store in app state so /ask can use it
        rag_state["index"]  = index
        rag_state["chunks"] = loaded_chunks
        rag_state["ready"]  = True

        logger.info("PDF processed and ready for questions")

        return {
            "message":      "PDF processed successfully",
            "chunks_count": len(chunks)
        }

    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process PDF: {str(e)}"
        )


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Receives a question + conversation history, returns a grounded answer.

    Flow:
    1. Check PDF is loaded
    2. Embed question
    3. Search FAISS
    4. Generate answer with LLM (passing history for memory)
    5. Return response
    """

    if not rag_state["ready"]:
        raise HTTPException(
            status_code=400,
            detail="No PDF uploaded yet. Please upload a PDF first."
        )

    logger.info(f"Question received: '{request.question}'")

    try:
        # Embed the question
        question_vector = embed_chunks([request.question])[0]

        # Search FAISS for relevant chunks
        relevant_chunks = search(
            query_vector=question_vector,
            index=rag_state["index"],
            chunks=rag_state["chunks"],
            top_k=5
        )

        # Generate answer — pass history for conversation memory
        answer = generate_answer(
            request.question,
            relevant_chunks,
            history=request.history
            # history contains all previous exchanges
            # LLM uses this to understand follow-up questions
        )

        return AnswerResponse(
            question=request.question,
            answer=answer
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        raise HTTPException(
            status_code=500,
            detail="Something went wrong. Please try again."
        )


@app.get("/health")
async def health_check():
    """Health check — used by deployment platforms to monitor the app."""
    return {
        "status":     "ok",
        "pdf_loaded": rag_state["ready"]
    }