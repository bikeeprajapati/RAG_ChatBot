# ◈ DocMind — Chat with Any PDF

A production-ready RAG (Retrieval-Augmented Generation) chatbot that lets you upload any PDF and have an intelligent conversation with it. Built from scratch without frameworks — every component is implemented manually to demonstrate deep understanding of the underlying architecture.

![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?style=flat-square)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-orange?style=flat-square)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Inference_API-yellow?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## What is RAG?

RAG (Retrieval-Augmented Generation) solves a fundamental problem with LLMs — they don't know your documents. Instead of fine-tuning a model (expensive) or dumping an entire PDF into a prompt (impractical), RAG:

1. **Splits** your document into small chunks
2. **Embeds** each chunk into a vector (numbers that encode meaning)
3. **Stores** those vectors in a searchable index
4. When you ask a question — **retrieves** only the relevant chunks
5. **Feeds** those chunks + your question to an LLM to generate a grounded answer

The result: accurate, source-grounded answers without hallucination.

---

## Architecture

```
PDF Upload
    │
    ▼
┌─────────────┐     ┌──────────────────┐     ┌─────────────┐
│  PDF Loader │────▶│  Text Splitter   │────▶│  Embedder   │
│  (PyMuPDF)  │     │  (500 char/50    │     │  (MiniLM    │
│             │     │   overlap chunks)│     │   via HF)   │
└─────────────┘     └──────────────────┘     └──────┬──────┘
                                                     │
                                                     ▼
                                             ┌─────────────┐
                                             │  FAISS      │
                                             │  Vector     │
                                             │  Store      │
                                             └──────┬──────┘
                                                    │
User Question                                       │
    │                                               │
    ▼                                               │
┌─────────────┐     ┌──────────────────┐            │
│  Embedder   │────▶│  Similarity      │◀───────────┘
│  (same      │     │  Search (top-5)  │
│   model)    │     └────────┬─────────┘
└─────────────┘              │
                             ▼
                    ┌─────────────────┐     ┌─────────────┐
                    │  Prompt Builder │────▶│  Mistral LLM│
                    │  (context +     │     │  (via HF    │
                    │   history)      │     │   Router)   │
                    └─────────────────┘     └──────┬──────┘
                                                   │
                                                   ▼
                                              Answer ✓
```

---

## Project Structure

```
DocMind/
├── src/
│   ├── chunking.py        # PDF loading + text splitting
│   ├── embedder.py        # HF Inference API embeddings
│   ├── vector_store.py    # FAISS index build, save, load, search
│   ├── retriever.py       # Pipeline coordinator
│   └── chatbot.py         # LLM answer generation with memory
├── frontend/
│   ├── index.html         # Chat UI
│   ├── style.css          # Dark editorial design
│   └── app.js             # API communication + state management
├── data/                  # Uploaded PDFs + FAISS index (git-ignored)
├── main.py                # FastAPI application + endpoints
├── requirements.txt       # Python dependencies
├── .env.example           # Environment variable template
└── README.md
```

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| **Backend** | FastAPI | Async, auto-validation, auto-docs |
| **PDF Parsing** | PyMuPDF | Fast, reliable text extraction |
| **Embeddings** | `all-MiniLM-L6-v2` via HF API | Free, no download, 384-dim vectors |
| **Vector Store** | FAISS (IndexFlatL2) | Exact similarity search, no setup |
| **LLM** | `Arch-Router-1.5B` via HF Router | Free tier, OpenAI-compatible API |
| **Frontend** | Vanilla HTML/CSS/JS | No build step, fast, lightweight |
| **Deployment** | Render | Free tier, simple git-based deploy |

---

## Features

- **Upload any PDF** — drag and drop or click to browse
- **Instant processing** — chunks, embeds, and indexes in seconds
- **Conversation memory** — follow-up questions work naturally
- **Hallucination prevention** — LLM is strictly grounded to document context
- **Replace document** — swap PDFs without restarting the server
- **Modern UI** — dark editorial design, animated, responsive
- **API docs** — auto-generated at `/docs` via FastAPI

---

## Getting Started

### Prerequisites

- Python 3.10+
- A free [HuggingFace](https://huggingface.co) account and API token

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/RAG_ChatBot.git
cd RAG_ChatBot
```

**2. Create and activate virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
.venv\Scripts\activate           # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up environment variables**
```bash
cp .env.example .env
```

Edit `.env` and add your HuggingFace token:
```
HUGGINGFACE_API_TOKEN=hf_your_token_here
```

Get your token at: `huggingface.co → Settings → Access Tokens`

**5. Run the server**
```bash
uvicorn main:app --reload
```

**6. Open your browser**
```
http://localhost:8000
```

---

## Usage

1. **Upload a PDF** — drag it into the upload zone or click to browse
2. **Wait for processing** — the app chunks, embeds, and indexes your document
3. **Ask questions** — type naturally, follow-ups work automatically
4. **Replace document** — click "Replace document" to chat with a different PDF

### Example Questions
```
What is this document about?
Summarize the key points
What are the main findings?
Tell me more about [specific topic]
What does the author recommend?
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the chat UI |
| `POST` | `/upload` | Upload and process a PDF |
| `POST` | `/ask` | Ask a question, get an answer |
| `GET` | `/health` | Server health + PDF status |
| `GET` | `/docs` | Auto-generated API documentation |

### Example API Usage

```bash
# Upload a PDF
curl -X POST http://localhost:8000/upload \
  -F "file=@document.pdf"

# Ask a question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?", "history": []}'
```

---

## How Each Component Works

### `src/chunking.py`
Extracts text from PDFs using PyMuPDF and splits it into 300-character overlapping chunks with 100-character overlap. Overlap prevents information loss at chunk boundaries.

### `src/embedder.py`
Sends text to HuggingFace's `all-MiniLM-L6-v2` model via their Inference API. Returns 384-dimensional vectors where similar meanings produce similar numbers — enabling semantic search.

### `src/vector_store.py`
Uses FAISS `IndexFlatL2` to store and search vectors. Saves the index and original chunks separately to disk — vectors for search, text for retrieval. Loads once at startup, searches in memory.

### `src/retriever.py`
Coordinates the pipeline: embeds the question, searches FAISS for top-5 most relevant chunks, returns readable text to the chatbot.

### `src/chatbot.py`
Builds a structured prompt with system rules + conversation history + retrieved context + current question. Sends to Mistral via HuggingFace's OpenAI-compatible router. Strict system prompt prevents hallucination.

### `main.py`
FastAPI application with async endpoints. Validates requests via Pydantic models. Stores FAISS index in memory between requests. Serves the frontend as static files.

---

## Design Decisions

**Why no LangChain?**
Built from scratch intentionally — every component is explicit and understandable. LangChain abstracts the exact mechanics this project was built to learn.

**Why FAISS over a vector database?**
FAISS runs locally with zero setup and zero cost. For a single-user app with documents up to a few hundred pages, it's the right tool. Production systems with millions of documents would use Pinecone, Weaviate, or pgvector.

**Why `IndexFlatL2`?**
Exact search — every vector is compared, results are always accurate. For learning and small datasets this is ideal. At millions of vectors you'd switch to `IndexIVFFlat` for approximate search with much better speed.

**Why conversation history is capped at 12 messages?**
Each message adds tokens to the prompt. Too much history makes the prompt too long, increasing latency and API cost. 6 exchanges (12 messages) is enough context for natural conversation.

---

## Environment Variables

| Variable | Description | Required |
|---|---|---|
| `HUGGINGFACE_API_TOKEN` | HuggingFace API token for embeddings + LLM | Yes |

---

## Requirements

```
fastapi
uvicorn
python-multipart
pymupdf
faiss-cpu
numpy
openai
huggingface-hub
python-dotenv
```

---

## Roadmap

- [ ] Multi-document support
- [ ] Query rewriting for better retrieval
- [ ] Source highlighting — show which chunk answered the question
- [ ] Chat history persistence across sessions
- [ ] LangChain refactor (next project)
- [ ] Deploy to Render

---

## What I Learned

Building this project from scratch taught me:

- How RAG works at every layer — not just as a concept but as running code
- Why chunking strategy directly affects answer quality
- How embeddings encode meaning as numbers and enable semantic search
- Why the same embedding model must be used for both documents and queries
- How FAISS indexes vectors for fast similarity search
- Why prompt engineering is the difference between grounded and hallucinated answers
- How FastAPI's async model handles concurrent requests efficiently
- How conversation history enables follow-up questions in stateless HTTP

---

## Author

**Bikee Prajapati**
- GitHub: [github.com/bikeeprajapati](https://github.com/bikeeprajapati)
- LinkedIn: [linkedin.com/in/bikee-prajapati9898](https://linkedin.com/in/bikee-prajapati9898)
- Email: bikeeprajapati1@gmail.com

---

## License

MIT License — free to use, modify, and distribute.