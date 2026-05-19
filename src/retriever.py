import logging
import numpy as np

from embedder import embed_chunks
from vector_store import load_vector_store, search

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
logger = logging.getLogger(__name__)

# CONSTANTS 
DEFAULT_TOP_K = 3


# LOAD ONCE AT STARTUP 
# We load the index once when this module is imported
# not on every question — loading is slow, searching is fast
# In a real app this runs once when the server starts
index, chunks = load_vector_store()


#  CORE FUNCTION 
def retrieve(question: str, top_k: int = DEFAULT_TOP_K) -> list[str]:
    """
    Full retrieval pipeline — question in, relevant chunks out.

    Args:
        question: Raw question string from the user
        top_k:    Number of chunks to return (default: 3)

    Returns:
        List of most relevant text chunks, ranked best-first
    """

    if not question.strip():
        # Guard clause — reject empty questions immediately
        # Never let bad input reach the embedding model or FAISS
        raise ValueError("Question cannot be empty.")

    logger.info(f"Retrieving for: '{question}'")

    # Step 1 — embed the question into a vector
    # We wrap in a list because embed_chunks expects a list of strings
    # [0] gets the first (and only) vector back out
    question_vector = embed_chunks([question])[0]

    # Step 2 — search FAISS for the closest chunks
    # Pass the pre-loaded index and chunks — no disk reads on every call
    relevant_chunks = search(
        query_vector=question_vector,
        index=index,
        chunks=chunks,
        top_k=top_k
    )

    logger.info(f"Found {len(relevant_chunks)} relevant chunks")
    return relevant_chunks


# HELPER: format for LLM 
def format_context(chunks: list[str]) -> str:
    """
    Joins retrieved chunks into a single formatted string.
    This is what gets inserted into the LLM prompt in Stage 5.

    We number each chunk so the LLM understands there are
    multiple sources and can reason across them.
    """
    return "\n\n".join(
        f"[Chunk {i+1}]:\n{chunk.strip()}"
        for i, chunk in enumerate(chunks)
    )


#  ENTRY POINT 
if __name__ == "__main__":
    import sys
    sys.path.append("..")

    # Test with three different questions
    # Each should pull different parts of your resume
    test_questions = [
        "What are Bikee's technical skills?",
        "What projects has Bikee built?",
        "Where is Bikee from and how to contact him?"
    ]

    for question in test_questions:
        print(f"\n{'='*55}")
        print(f"Q: {question}")
        print('='*55)

        results  = retrieve(question)
        context  = format_context(results)
        print(context)