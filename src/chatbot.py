import os
import logging
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
logger = logging.getLogger(__name__)

#  CONSTANTS 
LLM_MODEL   = "mistralai/Mistral-7B-Instruct-v0.3"
# Mistral-7B: powerful open-source LLM, free on HF inference API
# "Instruct" version = fine-tuned to follow instructions and answer questions
# This is what reads your context and generates the final answer

MAX_TOKENS  = 512
# Maximum length of the generated answer
# 512 tokens ≈ ~380 words — enough for a detailed answer

TEMPERATURE = 0.3
# Controls randomness in generation
# 0.0 = deterministic, always same answer
# 1.0 = very creative, unpredictable
# 0.3 = mostly factual with slight natural variation — good for Q&A


#  SETUP 
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN missing. Add it to your .env file.")

# Same InferenceClient as embedder but now we use it for text generation
# Created once at module level — not inside functions
client = InferenceClient(
    provider="hf-inference",
    api_key=HF_TOKEN
)


#  PROMPT BUILDER   
def build_prompt(question: str, context_chunks: list[str]) -> str:
    """
    Assembles the final prompt sent to the LLM.

    RAG prompt structure — always three parts:
    1. System instruction — tell the LLM its role and rules
    2. Context          — the retrieved chunks from your document
    3. Question         — what the user actually asked

    Why this order: LLMs perform better when they read the
    context BEFORE the question — they know what to look for.
    """

    # Join chunks with clear separators so LLM sees them as distinct sources
    context = "\n\n".join(
        f"[Source {i+1}]:\n{chunk.strip()}"
        for i, chunk in enumerate(context_chunks)
    )

    # This is a standard RAG prompt template
    # "Only use the context below" = prevents hallucination
    # The LLM won't invent facts — it sticks to your document
    prompt = f"""You are a helpful assistant that answers questions about Bikee Prajapati's resume.
Use ONLY the context provided below to answer the question.
If the answer is not in the context, say "I don't have that information in the resume."
Do not make up or assume any information.

Context:
{context}

Question: {question}

Answer:"""

    return prompt


#  CORE FUNCTION        
def generate_answer(question: str, context_chunks: list[str]) -> str:
    """
    Generates a grounded answer using retrieved context.

    Args:
        question:       Raw question from the user
        context_chunks: Retrieved chunks from vector store

    Returns:
        Generated answer string
    """

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if not context_chunks:
        raise ValueError("No context chunks provided.")

    prompt = build_prompt(question, context_chunks)
    logger.info(f"Sending prompt to {LLM_MODEL}...")

    response = client.text_generation(
        prompt,
        model=MAX_TOKENS,
        max_new_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        do_sample=True,
        # do_sample=True → use temperature-based sampling
        # do_sample=False → always pick highest probability token (greedy)
        # We use True so answers feel natural, not robotic
    )

    answer = response.strip()
    logger.info(f"Generated answer ({len(answer)} chars)")
    return answer


#  ENTRY POINT 
if __name__ == "__main__":
    import sys
    sys.path.append("..")

    from chunking import load_pdf, chunk_text
    from embedder import embed_chunks
    from vector_store import load_vector_store, search

    index, chunks = load_vector_store()

    question        = "What programming languages does Bikee know?"
    question_vector = embed_chunks([question])[0]
    context_chunks  = search(question_vector, index, chunks)

    print(f"\nQ: {question}")
    print(f"\nA: {generate_answer(question, context_chunks)}")