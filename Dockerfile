# ── BASE IMAGE ────────────────────────────────────────────────────────
# Official Python 3.12 slim image — slim = smaller, no unnecessary tools
# Always pin the version — "latest" can break your build unexpectedly
FROM python:3.12-slim

# ── METADATA ──────────────────────────────────────────────────────────
# Labels are optional but professional — documents who made this image
LABEL maintainer="Bikee Prajapati <bikeeprajapati1@gmail.com>"
LABEL description="DocMind — RAG chatbot for PDF documents"

# ── WORKING DIRECTORY ─────────────────────────────────────────────────
# All commands from here run inside /app inside the container
# Think of it as cd /app — except the directory gets created automatically
WORKDIR /app

# ── SYSTEM DEPENDENCIES ───────────────────────────────────────────────
# PyMuPDF needs these system libraries to parse PDFs
# We install them before Python packages
RUN apt-get update && apt-get install -y \
    libmupdf-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*
# rm -rf /var/lib/apt/lists/* removes the apt cache
# Keeps the image smaller — good practice in every Dockerfile

# ── PYTHON DEPENDENCIES ───────────────────────────────────────────────
# Copy requirements.txt FIRST — before copying the rest of your code
# Why: Docker caches each step. If requirements.txt didn't change,
# Docker reuses the cached layer and skips pip install (much faster builds)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
# --no-cache-dir = don't store pip's download cache inside the image
# Keeps image size smaller

# ── APPLICATION CODE ──────────────────────────────────────────────────
# Copy everything else AFTER installing dependencies
# This way code changes don't invalidate the pip install cache
COPY . .

# ── DATA DIRECTORY ────────────────────────────────────────────────────
# Create the data/ folder inside the container
# Uploaded PDFs and FAISS index get saved here at runtime
RUN mkdir -p data

# ── PORT ──────────────────────────────────────────────────────────────
# Tell Docker this container listens on port 8000
# EXPOSE is documentation — it doesn't actually open the port
# You open the port when running the container with -p flag
EXPOSE 8000

# ── STARTUP COMMAND ───────────────────────────────────────────────────
# This runs when the container starts
# --host 0.0.0.0 = listen on all network interfaces inside the container
#                  Without this, the app only listens inside the container
#                  and you can't reach it from your browser
# --port 8000    = the port to listen on
# No --reload    = reload is for development only, not production
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
