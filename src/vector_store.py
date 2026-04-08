
import faiss
import numpy as np
import pickle
import os


# BUILD AND SAVE 
def build_vector_store(chunks: list[str], embeddings: np.ndarray, save_dir: str = "data"):
    """
    Takes chunks + their vectors → builds FAISS index → saves both to disk.

    Why save both?
    - FAISS index stores the VECTORS (for searching)
    - chunks list stores the ORIGINAL TEXT (so we can return readable text)
    We need both because FAISS only knows about numbers, not the original text.
    """

    # embeddings must be float32 — FAISS requires this specific data type
    # float32 = 32-bit floating point number (less precise than float64 but faster)
    embeddings = embeddings.astype(np.float32)

    # Get the size of each vector — in our case 384
    # embeddings.shape = (num_chunks, 384), so shape[1] = 384
    dimension = embeddings.shape[1]

    # IndexFlatL2 = the simplest FAISS index type
    # "Flat" = stores all vectors as-is, no compression
    # "L2" = uses L2 distance (Euclidean distance) to measure similarity
    # For learning this is perfect — simple, exact, no approximation
    index = faiss.IndexFlatL2(dimension)

    # Add all our vectors into the index
    # Now FAISS knows about all 8 chunks from your resume
    index.add(embeddings)

    print(f"✓ FAISS index built with {index.ntotal} vectors")

    # ── SAVE TO DISK ──────────────────────────────────────────────────
    index_path = os.path.join(save_dir, "index.faiss")
    chunks_path = os.path.join(save_dir, "chunks.pkl")

    # faiss.write_index() saves the vector index to a binary file
    # This is FAISS's own format — only readable by FAISS
    faiss.write_index(index, index_path)

    # pickle.dump() saves our chunks list to a file
    # "wb" = write binary — pickle files are binary, not plain text
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)

    print(f"✓ Saved index  → {index_path}")
    print(f"✓ Saved chunks → {chunks_path}")


# ── LOAD FROM DISK ────────────────────────────────────────────────────
def load_vector_store(save_dir: str = "data"):
    """
    Loads the FAISS index and chunks from disk.
    Call this when you want to search — no need to rebuild every time.
    """

    index_path = os.path.join(save_dir, "index.faiss")
    chunks_path = os.path.join(save_dir, "chunks.pkl")

    # faiss.read_index() reads the binary file back into a FAISS index object
    index = faiss.read_index(index_path)

    # pickle.load() reconstructs our Python list of chunks from the file
    # "rb" = read binary
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)

    print(f"✓ Loaded index with {index.ntotal} vectors")
    print(f"✓ Loaded {len(chunks)} chunks")
    return index, chunks


# ── SEARCH ────────────────────────────────────────────────────────────
def search(query_vector: np.ndarray, index, chunks: list[str], top_k: int = 3):
    """
    Given a question's vector → find the top_k most similar chunks.

    top_k = how many chunks to return
    We use 3 by default — enough context, not too much noise
    """

    # FAISS expects float32 and a 2D array even for a single vector
    # reshape(1, -1) turns [384 numbers] into [[384 numbers]]
    # The 1 means "1 query", -1 means "figure out the rest automatically"
    query_vector = query_vector.astype(np.float32).reshape(1, -1)

    # index.search() is the core FAISS operation
    # Returns two arrays:
    # distances = how far each result is (lower = more similar for L2)
    # indices   = which position in our chunks list each result is at
    distances, indices = index.search(query_vector, top_k)

    print(f"\n🔍 Top {top_k} most relevant chunks:")

    results = []
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        # distances[0] and indices[0] because FAISS returns 2D arrays
        # (designed to handle multiple queries at once)
        chunk = chunks[idx]
        print(f"\n--- Rank {rank+1} (distance: {dist:.4f}) ---")
        print(chunk[:200] + "...")  # show first 200 chars as preview
        results.append(chunk)

    return results


# ── TEST ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.append("..")

    from chunking import load_pdf, chunk_text
    from embedder import embed_chunks, cosine_similarity

    # Step 1: Load and chunk the resume
    chunks = chunk_text(load_pdf("data/Bikee_Prajapati.pdf"))

    # Step 2: Embed all chunks
    embeddings = embed_chunks(chunks)

    # Step 3: Build and save the vector store
    build_vector_store(chunks, embeddings)

    # Step 4: Load it back (proving it saved correctly)
    index, loaded_chunks = load_vector_store()

    # Step 5: Search with a real question
    # First embed the question — same model, same vector space
    print("\n🧪 Test search: 'What are Bikee's technical skills?'")
    question = ["What are Bikee's technical skills?"]
    question_vector = embed_chunks(question)[0]
    # [0] because embed_chunks returns array of vectors — we only have one question

    # Search the index
    results = search(question_vector, index, loaded_chunks, top_k=3)