import os
import logging
from openai import OpenAI
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
logger = logging.getLogger(__name__)

# ── CONSTANTS ─────────────────────────────────────────────────────────
LLM_MODEL      = "katanemo/Arch-Router-1.5B:hf-inference"
# katanemo/Arch-Router-1.5B — small instruction-tuned model
# Works on HF free tier — no GPU, no cost for light usage
# ":hf-inference" tells the router which provider to use

MAX_NEW_TOKENS = 512
# Max tokens the LLM can generate in one response
# 512 ≈ 380 words — enough for a detailed answer

TEMPERATURE    = 0.2
# How creative/random the answer is
# 0.2 = mostly factual, stays close to context — right for Q&A


# ── SETUP ─────────────────────────────────────────────────────────────
load_dotenv()
# Reads .env file and loads HF_TOKEN into memory

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not HF_TOKEN:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN missing. Add it to your .env file.")

# We use OpenAI client pointing at HF's router endpoint
# HF router is OpenAI-compatible — same interface, HF models
# This is the correct way to call LLMs on HF free tier in 2025+
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN
)


# ── CORE FUNCTION ─────────────────────────────────────────────────────
def generate_answer(question: str, context_chunks: list[str]) -> str:
    """
    Generates a grounded answer from question + retrieved context.

    Args:
        question:       Raw question string from the user
        context_chunks: List of relevant text chunks from vector store

    Returns:
        Generated answer as a plain string
    """

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if not context_chunks:
        raise ValueError("No context chunks provided.")

    # Number each chunk so LLM treats them as separate sources
    # This helps the LLM reason across multiple passages
    context = "\n\n".join(
        f"[Source {i+1}]:\n{chunk.strip()}"
        for i, chunk in enumerate(context_chunks)
    )

    logger.info(f"Sending to LLM: '{question}'")

    # chat.completions.create() is the standard modern LLM interface
    # Same format as OpenAI — two roles:
    # "system" → instructions the LLM must follow
    # "user"   → the actual input (context + question)
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that answers questions "
                    "strictly based on the provided resume context. "
                    "If the answer is not in the context, say: "
                    "'I don't have that information in the resume.' "
                    "Never make up or assume anything not in the context. "
                    "Be concise and direct."
                )
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }
        ],
        max_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
    )
    # response structure (OpenAI format):
    # response.choices        → list of generated responses (we asked for 1)
    # response.choices[0]     → first response
    # .message                → the message object
    # .content                → the actual text string

    answer = response.choices[0].message.content.strip()
    logger.info(f"Answer generated — {len(answer)} chars")
    return answer


#  ENTRY POINT 
if __name__ == "__main__":
    import sys
    sys.path.append("..")

    from retriever import retrieve

    # Test with 3 different questions
    # Each hits a different part of the resume
    test_questions = [
        "What programming languages does Bikee know?",
        "What projects has Bikee built?",
        "What is Bikee's educational background?",
    ]

    for question in test_questions:
        print(f"\n{'='*55}")
        print(f"Q: {question}")
        print("="*55)

        context_chunks = retrieve(question)

        print("⏳ Generating answer...")
        answer = generate_answer(question, context_chunks)

        print(f"A: {answer}")