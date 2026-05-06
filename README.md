# ◈ DocMind — Chat with Any PDF

A production-ready RAG (Retrieval-Augmented Generation) chatbot that lets you upload any PDF and have an intelligent conversation with it. Built from scratch without frameworks — every component is implemented manually to demonstrate deep understanding of the underlying architecture.

![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?style=flat-square)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-orange?style=flat-square)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Inference_API-yellow?style=flat-square)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## Live Demo

> Coming soon — deploying on Render

---

## Run in 30 Seconds (Docker)

No Python, no pip install, no setup needed. Just Docker:

```bash
docker run -d \
  -p 8000:8000 \
  -e HUGGINGFACE_API_TOKEN=your_token_here \
  --name docmind \
  bikeeprajapati1/docmind:latest
```

Open your browser at:
```
http://localhost:8000
```

Get your free HuggingFace token at: [huggingface.co → Settings → Access Tokens](https://huggingface.co/settings/tokens)

---

## What is RAG?

RAG (Retrieval-Augmented Generation) solves a fundamental problem with LLMs — they don't know your documents. Instead of fine-tuning a model (expensive) or dumping an entire PDF into a prompt (impractical), RAG:

1. **Splits** your document into small overlapping chunks
2. **Embeds** each chunk into a vector — numbers that encode meaning
3. **Stores** those vectors in a searchable FAISS index
4. When you ask a question — **retrieves** only the relevant chunks
5. **Feeds** those chunks + your question to an LLM to generate a grounded answer

The result: accurate, source-grounded answers without hallucination.

---

## Architecture

```
PDF Upload
    │
    ▼
┌─────────────┐     ┌──────────────────┐     ┌──────────────┐
│  PDF Loader │────▶│  Text Splitter   │────▶│   Embedder   │
│  (PyMuPDF)  │     │  300 char chunks │     │  MiniLM-L6   │
│             │     │  100 char overlap│     │  via HF API  │
└─────────────┘     └──────────────────┘     └──────┬───────┘
                                                     │
                                                     ▼
                                             ┌──────────────┐
                                             │    FAISS     │
                                             │ Vector Store │
                                             │ (persisted)  │
                                             └──────┬───────┘
                                                    │
User Question                                       │
    │                                               │
    ▼                                               │
┌─────────────┐     ┌──────────────────┐            │
│   Embedder  │────▶│ Similarity Search│◀───────────┘
│ (same model)│     │    top-5 chunks  │
└─────────────┘     └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐     ┌──────────────┐
                    │  Prompt Builder  │────▶│  Mistral LLM │
                    │ context+history  │     │  HF Router   │
                    └──────────────────┘     └──────┬───────┘
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
│   └── app.js             # API communication + conversation state
├── data/                  # Uploaded PDFs + FAISS index (git-ignored)
├── main.py                # FastAPI application + all endpoints
├── Dockerfile             # Container definition
├── .dockerignore          # Files excluded from Docker image
├── requirements.txt       # Python dependencies
├── .env.example           # Environment variable template
└── README.md
```

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| **Backend** | FastAPI | Async, auto-validation, auto-docs at `/docs` |
| **PDF Parsing** | PyMuPDF | Fast, reliable text extraction |
| **Embeddings** | `all-MiniLM-L6-v2` via HF API | Free, no download, 384-dim vectors |
| **Vector Store** | FAISS `IndexFlatL2` | Exact similarity search, zero setup |
| **LLM** | `Arch-Router-1.5B` via HF Router | Free tier, OpenAI-compatible API |
| **Frontend** | Vanilla HTML / CSS / JS | No build step, fast, lightweight |
| **Container** | Docker | Consistent environment everywhere |
| **Registry** | Docker Hub | Public image distribution |

---

## Features

- **Upload any PDF** — drag and drop or click to browse
- **Instant processing** — chunks, embeds, and indexes in seconds
- **Conversation memory** — follow-up questions work naturally
- **Hallucination prevention** — LLM strictly grounded to document context
- **Replace document** — swap PDFs without restarting the server
- **Modern dark UI** — animated, responsive, no generic AI look
- **Dockerized** — runs anywhere with one command
- **API docs** — auto-generated at `/docs` via FastAPI

---

## Getting Started (Without Docker)

### Prerequisites

- Python 3.10+
- A free [HuggingFace](https://huggingface.co) account and API token

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/bikeeprajapati/RAG_ChatBot.git
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

Edit `.env`:
```
HUGGINGFACE_API_TOKEN=hf_your_token_here
```

**5. Run the server**
```bash
uvicorn main:app --reload
```

**6. Open your browser**
```
http://localhost:8000
```

---

## Getting Started (With Docker)

**Pull and run from Docker Hub:**
```bash
docker run -d \
  -p 8000:8000 \
  -e HUGGINGFACE_API_TOKEN=your_token_here \
  --name docmind \
  bikeeprajapati1/docmind:latest
```

**Or build locally:**
```bash
git clone https://github.com/bikeeprajapati/RAG_ChatBot.git
cd RAG_ChatBot
docker build -t docmind .
docker run -d \
  -p 8000:8000 \
  -e HUGGINGFACE_API_TOKEN=your_token_here \
  --name docmind \
  docmind
```

**Useful Docker commands:**
```bash
docker logs -f docmind        # follow live logs
docker stop docmind           # stop the container
docker start docmind          # start it again
docker rm docmind             # remove the container
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the chat UI |
| `POST` | `/upload` | Upload and process a PDF |
| `POST` | `/ask` | Ask a question, get an answer |
| `GET` | `/health` | Server health + PDF loaded status |
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

# Health check
curl http://localhost:8000/health
```

---

## How Each Component Works

### `src/chunking.py`
Extracts text from PDFs using PyMuPDF and splits into 300-character chunks with 100-character overlap. Overlap prevents information loss at chunk boundaries — a sentence cut in half is still captured fully in at least one chunk.

### `src/embedder.py`
Sends text to HuggingFace's `all-MiniLM-L6-v2` via their Inference API. Returns 384-dimensional vectors where similar meanings produce similar numbers — enabling semantic search instead of keyword matching.

### `src/vector_store.py`
Uses FAISS `IndexFlatL2` to store and search vectors. Saves index (numbers) and chunks (text) separately — FAISS can only store numbers, we need the text to return readable answers. Loads once at startup, searches entirely in memory.

### `src/retriever.py`
Coordinates the pipeline: embeds the question using the same model as the documents, searches FAISS for top-5 most semantically similar chunks, returns readable text to the chatbot.

### `src/chatbot.py`
Builds structured messages: system rules + conversation history + retrieved context + current question. Sends to Mistral via HuggingFace's OpenAI-compatible router. Strict system prompt prevents hallucination. History capped at 12 messages to control token cost.

### `main.py`
FastAPI application with async endpoints. Pydantic models validate every request automatically. Stores FAISS index in memory between requests. Serves frontend as static files. Health endpoint for deployment monitoring.

---

## Environment Variables

| Variable | Description | Required |
|---|---|---|
| `HUGGINGFACE_API_TOKEN` | HuggingFace API token for embeddings + LLM | Yes |

---

## Design Decisions

**Why no LangChain?**
Built from scratch intentionally — every component is explicit and understandable. LangChain abstracts the exact mechanics this project was built to learn. A LangChain version is planned as a follow-up project.

**Why FAISS over a vector database?**
FAISS runs locally with zero setup and zero cost. For a single-user app with documents up to a few hundred pages it's the right tool. Production systems with millions of documents would use Pinecone, Weaviate, or pgvector.

**Why `IndexFlatL2`?**
Exact search — every vector is compared, results are always accurate. For learning and small datasets this is ideal. At millions of vectors you'd switch to `IndexIVFFlat` for approximate but much faster search.

**Why conversation history is capped at 12 messages?**
Each message adds tokens to the prompt. Too much history increases latency and API cost. 6 exchanges (12 messages) is enough context for natural conversation flow.

**Why OpenAI client pointing at HF router?**
HuggingFace's router is OpenAI-compatible — same interface, free models. The same code works with OpenAI, HuggingFace, or any OpenAI-compatible provider. Zero vendor lock-in.

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

- [ ] Deploy to Render — publicly accessible URL
- [ ] Multi-document support
- [ ] Query rewriting for better retrieval on vague questions
- [ ] Source highlighting — show exactly which chunk answered the question
- [ ] Chat history persistence across sessions
- [ ] Rebuild with LangChain (next project)

---

## What I Learned Building This

- How RAG works at every layer — not just as a concept but as running code
- Why chunking strategy directly affects retrieval quality
- How embeddings encode meaning as numbers and enable semantic search
- Why the same embedding model must be used for both documents and queries
- How FAISS indexes vectors for millisecond similarity search
- Why prompt engineering is the difference between grounded and hallucinated answers
- How FastAPI's async model handles concurrent requests without blocking
- How conversation history enables follow-up questions in stateless HTTP
- How Docker packages an entire environment into a portable container
- How to push images to Docker Hub for public distribution

---

## Docker Hub

Public image available at:

```bash
docker pull bikeeprajapati1/docmind:latest
```

[hub.docker.com/r/bikeeprajapati1/docmind](https://hub.docker.com/r/bikeeprajapati1/docmind)

---

## Author

**Bikee Prajapati**
- GitHub: [github.com/bikeeprajapati](https://github.com/bikeeprajapati)
- LinkedIn: [linkedin.com/in/bikee-prajapati9898](https://linkedin.com/in/bikee-prajapati9898)
- Email: bikeeprajapati1@gmail.com
- Docker Hub: [hub.docker.com/u/bikeeprajapati1](https://hub.docker.com/u/bikeeprajapati1)

---

## License

MIT License — free to use, modify, and distribute.