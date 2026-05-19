# RESPONSIBILITY: Generate a grounded answer from question + context.
# Now supports conversation history — understands follow-up questions.
#
# Input:  question (str) + context_chunks (list[str]) + history (list)
# Output: answer (str)
# ─────────────────────────────────────────────────────────────────────

import os
import logging
from openai import OpenAI
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
logger = logging.getLogger(__name__)

# ── CONSTANTS ─────────────────────────────────────────────────────────
LLM_MODEL      = "katanemo/Arch-Router-1.5B:hf-inference"
MAX_NEW_TOKENS = 512
TEMPERATURE    = 0.0
# 0.0 = fully deterministic — no creativity, pure factual mode
# Right choice for document Q&A where accuracy matters most


# ── SETUP ─────────────────────────────────────────────────────────────
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN missing. Add it to your .env file.")

# OpenAI client pointing at HF's router
# HF router is OpenAI-compatible — same interface, free tier models
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN
)


# ── CORE FUNCTION ─────────────────────────────────────────────────────
def generate_answer(
    question: str,
    context_chunks: list[str],
    history: list[dict] = []
) -> str:
    """
    Generates a grounded answer from question + context + conversation history.

    Args:
        question:       Raw question string from the user
        context_chunks: Retrieved chunks from vector store
        history:        Previous conversation messages
                        Format: [{"role": "user", "content": "..."},
                                 {"role": "assistant", "content": "..."}]

    Returns:
        Generated answer as a plain string
    """

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if not context_chunks:
        raise ValueError("No context chunks provided.")

    # Number each chunk so LLM treats them as separate sources
    context = "\n\n".join(
        f"[Source {i+1}]:\n{chunk.strip()}"
        for i, chunk in enumerate(context_chunks)
    )

    logger.info(f"Sending to LLM: '{question}' | history: {len(history)} messages")

    # ── BUILD MESSAGES ────────────────────────────────────────────────
    # message order always:
    # 1. system  → rules + context
    # 2. history → previous exchanges (gives memory)
    # 3. user    → current question
    messages = [
        {
            "role": "system",
            "content": (
                "You are a document assistant. Your ONLY job is to answer "
                "questions using the exact information provided in the context below."
                "\n\nSTRICT RULES:"
                "\n- ONLY use facts that are explicitly stated in the context"
                "\n- If the context does not contain the answer, respond with exactly: "
                "'The document does not contain information about this.'"
                "\n- Do NOT use your own knowledge"
                "\n- Do NOT make assumptions or inferences beyond what is written"
                "\n- Do NOT say things like 'likely', 'probably', or 'typically'"
                "\n- For follow-up questions like 'more', 'continue', or 'tell me more', "
                "expand on your previous answer using only the context"
                "\n- Be concise and direct"
                f"\n\nContext:\n{context}"
            )
        }
    ]

    # Append conversation history
    # This is what makes follow-up questions work
    # The LLM can see what was asked and answered before
    for msg in history:
        messages.append(msg)

    # Append current question as the final user message
    messages.append({"role": "user", "content": question})

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        max_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
    )

    answer = response.choices[0].message.content.strip()
    logger.info(f"Answer generated — {len(answer)} chars")
    return answer


# ── ENTRY POINT ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.append("..")

    from retriever import retrieve

    # Test conversation with follow-up
    questions = [
        ("What projects has Bikee built?", []),
    ]

    history = []
    for question, _ in questions:
        print(f"\nQ: {question}")
        chunks = retrieve(question)
        answer = generate_answer(question, chunks, history)
        print(f"A: {answer}")
        history.append({"role": "user",      "content": question})
        history.append({"role": "assistant", "content": answer})