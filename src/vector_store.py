import os
import faiss
import pickle
import logging
import numpy as np

# Industry practice: use logging, not print()
# logging gives you timestamps, severity levels, and can write to files
# In production you'd set this from a config — for now INFO level is fine
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
logger = logging.getLogger(__name__)
# __name__ = the module's own name ("vector_store")
# This way log messages show which file they came from


#CONSTANTS 
# Constants at the top, not buried in functions
# Makes them easy to find and change without hunting through code
INDEX_FILENAME  = "index.faiss"
CHUNKS_FILENAME = "chunks.pkl"
DEFAULT_TOP_K   = 3


# BUILD 
def build_vector_store(
    chunks: list[str],
    embeddings: np.ndarray,
    save_dir: str = "data"
) -> None:
    """
    Converts embeddings into a searchable FAISS index and saves to disk.

    Args:
        chunks:     Original text chunks — saved separately so we can
                    return readable text after search
        embeddings: Numpy array of shape (num_chunks, 384)
        save_dir:   Folder to save files into (default: data/)
    """

    # FAISS strictly requires float32
    # Your numpy array might be float64 by default — this ensures compatibility
    embeddings = embeddings.astype(np.float32)

    # shape[1] = number of columns = 384 (vector dimension)
    # We read this dynamically instead of hardcoding 384
    # because if you ever swap models, this still works
    dimension = embeddings.shape[1]

    # IndexFlatL2 = exact search using Euclidean (L2) distance
    # "Flat"  → stores every vector as-is, no compression
    # "L2"    → measures straight-line distance between vectors
    # For learning: always start with Flat — it's exact and simple
    # For production at scale: swap to IndexIVFFlat for speed
    index = faiss.IndexFlatL2(dimension)

    # Load all vectors into the index
    # After this, FAISS knows about all your chunks
    index.add(embeddings)
    logger.info(f"Built FAISS index — {index.ntotal} vectors, dimension {dimension}")

    #persist to disk 
    os.makedirs(save_dir, exist_ok=True)
    # exist_ok=True → don't crash if the folder already exists

    index_path  = os.path.join(save_dir, INDEX_FILENAME)
    chunks_path = os.path.join(save_dir, CHUNKS_FILENAME)

    faiss.write_index(index, index_path)
    # FAISS has its own binary format — use its own writer, not open()

    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)
    # pickle serializes any Python object to bytes
    # "wb" = write binary — pickle output is binary, not plain text

    logger.info(f"Saved index  → {index_path}")
    logger.info(f"Saved chunks → {chunks_path}")


# LOAD 
def load_vector_store(save_dir: str = "data") -> tuple[faiss.Index, list[str]]:
    """
    Loads FAISS index and chunks from disk.

    Industry pattern: index once, load many times.
    Building the index is expensive — loading is cheap.
    In a real app this runs once at startup, not on every request.

    Returns:
        (index, chunks) tuple
    """

    index_path  = os.path.join(save_dir, INDEX_FILENAME)
    chunks_path = os.path.join(save_dir, CHUNKS_FILENAME)

    # Fail early with a clear message rather than a confusing FAISS error
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"No FAISS index found at '{index_path}'.\n"
            f"Run build_vector_store() first to create it."
        )

    index = faiss.read_index(index_path)

    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    # "rb" = read binary — matches the "wb" we used when saving

    logger.info(f"Loaded index — {index.ntotal} vectors")
    logger.info(f"Loaded {len(chunks)} chunks")

    return index, chunks


#  SEARCH 
def search(
    query_vector: np.ndarray,
    index: faiss.Index,
    chunks: list[str],
    top_k: int = DEFAULT_TOP_K
) -> list[str]:
    """
    Finds the top_k most semantically similar chunks to a query vector.

    Args:
        query_vector: The embedded question (384 numbers)
        index:        Loaded FAISS index
        chunks:       Original text chunks (parallel to the index)
        top_k:        How many results to return (default: 3)

    Returns:
        List of the most relevant text chunks, ranked best-first
    """

    # FAISS expects shape (n_queries, dimension) — even for one query
    # reshape(1, -1) → turns (384,) into (1, 384)
    # The 1 = one query, -1 = numpy figures out the rest automatically
    query_vector = query_vector.astype(np.float32).reshape(1, -1)

    # index.search() returns two parallel arrays:
    # distances → L2 distance to each result (lower = more similar)
    # indices   → position of each result in our chunks list
    distances, indices = index.search(query_vector, top_k)

    # [0] because FAISS returns 2D arrays (supports multiple queries)
    # We only sent one query, so we take the first (and only) row
    results = []
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        chunk = chunks[idx]
        results.append(chunk)
        logger.info(f"Rank {rank+1} | distance: {dist:.4f} | preview: {chunk[:80]}...")

    return results


#  ENTRY POINT 
if __name__ == "__main__":
    import sys
    sys.path.append("..")

    from chunking import load_pdf, chunk_text
    from embedder import embed_chunks

    #  build phase (runs once) 
    logger.info("=== BUILD PHASE ===")
    chunks     = chunk_text(load_pdf("data/Bikee_Prajapati.pdf"))
    embeddings = embed_chunks(chunks)
    build_vector_store(chunks, embeddings)

    # ── query phase (runs on every question) 
    logger.info("=== QUERY PHASE ===")
    index, loaded_chunks = load_vector_store()

    question       = "What are Bikee's technical skills?"
    question_vector = embed_chunks([question])[0]

    logger.info(f"Query: '{question}'")
    results = search(question_vector, index, loaded_chunks)

    print("\n── Top Results ──")
    for i, result in enumerate(results):
        print(f"\n[{i+1}] {result[:300]}")
        print("─" * 50)