import sys
import os
import logging
import shutil

from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

sys.path.append("src")
# Tells Python to look inside src/ when importing our modules
# So we can write "from chatbot import generate_answer" cleanly

from chunking     import load_pdf, chunk_text
from embedder     import embed_chunks
from vector_store import build_vector_store, load_vector_store, search
from chatbot      import generate_answer

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
logger = logging.getLogger(__name__)


#  APP SETUP 
app = FastAPI(
    title="PDF RAG Chatbot",
    description="Upload a PDF and chat with it",
    version="1.0.0"
)

# Serve frontend files — HTML, CSS, JS — from frontend/ folder
# StaticFiles makes these files accessible to the browser
app.mount("/static", StaticFiles(directory="frontend"), name="static")


# ── CONSTANTS 
UPLOAD_DIR = "data"
# Where uploaded PDFs get saved before processing
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ── STATE 
# This holds the current user's vector index and chunks in memory
# Simple dict — works fine for a single-user app
# For multi-user: you'd use a database or session management
rag_state = {
    "index":  None,   # FAISS index — None until a PDF is uploaded
    "chunks": None,   # text chunks — None until a PDF is uploaded
    "ready":  False   # flag — True once PDF is processed and ready to query
}


# ── REQUEST / RESPONSE MODELS 
class QuestionRequest(BaseModel):
    question: str
    # Pydantic automatically:
    # 1. Parses incoming JSON → Python object
    # 2. Validates "question" exists and is a string
    # 3. Returns clear 422 error if validation fails
    # Never parse JSON manually — always use Pydantic models

class AnswerResponse(BaseModel):
    question: str
    answer:   str
    # Clean structured response — frontend knows exactly what fields to expect


# ── ROUTES ────────────────────────────────────────────────────────────
@app.get("/")
async def serve_frontend():
    """Serves the chat UI to the browser."""
    return FileResponse("frontend/index.html")


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Accepts a PDF upload, processes it through the full RAG pipeline,
    and stores the result in memory ready for questions.

    Flow:
    1. Save uploaded PDF to disk
    2. Extract and chunk text
    3. Embed chunks via HF API
    4. Build and save FAISS index
    5. Load index into memory
    6. Set rag_state ready = True
    """

    # Validate file type — only accept PDFs
    # .filename gives the original filename the user uploaded
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted."
        )

    # ── Save uploaded file to disk ────────────────────────────────────
    pdf_path = os.path.join(UPLOAD_DIR, "uploaded.pdf")
    # We always save as "uploaded.pdf" — overwrites previous upload
    # Simple approach for a single-user app

    with open(pdf_path, "wb") as f:
        content = await file.read()
        # await file.read() reads the entire uploaded file as bytes
        # "wb" = write binary — PDFs are binary files not text
        f.write(content)

    logger.info(f"PDF saved → {pdf_path} ({len(content)} bytes)")

    # ── Process through RAG pipeline ──────────────────────────────────
    try:
        # Step 1: Extract text and chunk
        logger.info("Chunking PDF...")
        text   = load_pdf(pdf_path)
        chunks = chunk_text(text)
        logger.info(f"Created {len(chunks)} chunks")

        # Step 2: Embed all chunks
        logger.info("Embedding chunks...")
        embeddings = embed_chunks(chunks)

        # Step 3: Build and save FAISS index
        logger.info("Building vector store...")
        build_vector_store(chunks, embeddings, save_dir=UPLOAD_DIR)

        # Step 4: Load into memory for querying
        index, loaded_chunks = load_vector_store(save_dir=UPLOAD_DIR)

        # Step 5: Store in app state so /ask can use it
        rag_state["index"]  = index
        rag_state["chunks"] = loaded_chunks
        rag_state["ready"]  = True
        # We store in rag_state dict so the /ask endpoint can access it
        # This is in-memory state — lives as long as the server runs
        # If server restarts, user needs to re-upload (acceptable for now)

        logger.info("PDF processed and ready for questions")

        return {
            "message":     "PDF processed successfully",
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
    Receives a question, retrieves relevant chunks, generates answer.

    Flow:
    1. Check PDF has been uploaded
    2. Embed the question
    3. Search FAISS for relevant chunks
    4. Generate answer with LLM
    5. Return structured response
    """

    # Guard — reject questions if no PDF has been processed yet
    if not rag_state["ready"]:
        raise HTTPException(
            status_code=400,
            detail="No PDF uploaded yet. Please upload a PDF first."
        )

    logger.info(f"Question received: '{request.question}'")

    try:
        # Step 1: Embed the question
        question_vector = embed_chunks([request.question])[0]
        # [request.question] → wrap in list (embed_chunks expects batch)
        # [0] → unpack the single vector back out

        # Step 2: Search FAISS for relevant chunks
        relevant_chunks = search(
            query_vector=question_vector,
            index=rag_state["index"],
            chunks=rag_state["chunks"],
            top_k=3
        )

        # Step 3: Generate answer
        answer = generate_answer(request.question, relevant_chunks)

        return AnswerResponse(
            question=request.question,
            answer=answer
        )

    except ValueError as e:
        # Bad input from user — 400 Bad Request
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        # Unexpected server error — log fully, return clean message
        logger.error(f"Error generating answer: {e}")
        raise HTTPException(
            status_code=500,
            detail="Something went wrong. Please try again."
        )


@app.get("/health")
async def health_check():
    """
    Confirms the server is alive and whether a PDF is loaded.
    Standard in every production API — deployment platforms ping this.
    Also tells the frontend whether the app is ready for questions.
    """
    return {
        "status": "ok",
        "pdf_loaded": rag_state["ready"]
    }