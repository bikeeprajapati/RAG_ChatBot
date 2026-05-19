import os
import numpy as np
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

if not HF_TOKEN:
    raise ValueError("HUGGINGFACE_API_TOKEN missing. Check your .env file.")
client = InferenceClient(
    provider="hf-inference",
    api_key=HF_TOKEN
)

# The model we use for embeddings — runs on HF's servers  
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


#CORE FUNCTION 
def embed_chunks(chunks: list[str]) -> np.ndarray:
    """
    Input:  list of text strings
    Output: numpy array of shape (num_chunks, 384)
            each row = one chunk encoded as 384 numbers
    """
    embeddings = client.feature_extraction(
        text=chunks,
        model=EMBEDDING_MODEL
    )

    # Convert to numpy array for fast math operations later
    embeddings = np.array(embeddings)
    return embeddings


#SIMILARITY HELPER 
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    How similar are two vectors in meaning?
    1.0 = identical meaning
    0.0 = completely unrelated

    Math: measures the angle between two vectors in 384D space
    Small angle = pointing same direction = similar meaning
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# TEST 
if __name__ == "__main__":
    import sys
    sys.path.append("..")
    # Adds parent folder to Python's search path
    # so we can import from loader.py / chunking.py

    from chunking import load_pdf, chunk_text

    chunks = chunk_text(load_pdf("data/Bikee_Prajapati.pdf"))
    embeddings = embed_chunks(chunks)

    score = cosine_similarity(embeddings[0], embeddings[1])
    print(f"\n📊 Similarity chunk1 ↔ chunk2: {score:.4f}")
    print("(From same resume → should be high, above 0.7)")

    print(f"\n🔢 First 10 numbers of chunk 1's vector:")
    print(embeddings[0][:10].round(4))
    print("\nThese numbers mean nothing visually — but mathematically")
    print("they place your text in 384D space. Similar text = nearby point.")