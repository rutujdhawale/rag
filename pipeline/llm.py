"""
llm.py

Dual-provider LLM setup for the Hybrid RAG pipeline:
  - Primary:  OpenRouter  (google/gemma-3-27b-it:free — fast, vision-capable, free)
  - Fallback: Groq        (llama-3.3-70b-versatile — fast free-tier inference)

Auto-fallback: if the primary call fails (rate limit / timeout / error),
the pipeline seamlessly retries on the fallback provider.

Usage:
    from pipeline.llm import get_llm, generate_answer

    llm = get_llm()
    answer = generate_answer("your question", retrieved_docs, llm=llm)
"""

import os
import logging
from typing import List, Optional, Union

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

from langchain_groq import ChatGroq

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemma-3-27b-it:free")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
DEFAULT_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.2))

if not OPENROUTER_API_KEY:
    logger.warning(
        "OPENROUTER_API_KEY not found. Set it in .env "
        "(get a free key at https://openrouter.ai/keys)."
    )
if not GROQ_API_KEY:
    logger.warning(
        "GROQ_API_KEY not found. Set it in .env "
        "(get a free key at https://console.groq.com/keys)."
    )

# ---------------------------------------------------------------------------
# RAG prompt — improved to not refuse on partial matches
# ---------------------------------------------------------------------------
RAG_PROMPT_TEMPLATE = """You are an expert AI assistant. Answer the user's question directly, clearly, and accurately based on the provided context.

Guidelines:
- Deliver a clear, concise, well-structured answer in clean Markdown (use concise bullet points or short paragraphs).
- Do NOT generate wide ASCII tables, meta-discussions, or redundant headings like "What's missing from the excerpts" or "Bottom line".
- Focus directly on facts, key insights, and actionable answers requested by the user.
- Synthesize information cleanly without repeating raw context phrases.
- If the context does not contain enough info, state what is known directly in 1-2 sentences.

Context:
{context}

Question: {question}

Answer:"""


# ---------------------------------------------------------------------------
# Provider builders
# ---------------------------------------------------------------------------
def get_openrouter_llm(
    model: str = OPENROUTER_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
) -> ChatOpenAI:
    """
    Build a ChatOpenAI instance pointing at OpenRouter's API.

    OpenRouter exposes an OpenAI-compatible endpoint, so we reuse
    langchain_openai.ChatOpenAI with a custom base_url.
    """
    if not OPENROUTER_API_KEY:
        raise EnvironmentError(
            "OPENROUTER_API_KEY is not set. Add it to your .env file."
        )

    logger.info(f"Initializing OpenRouter LLM (model='{model}')...")

    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=2048,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "http://localhost",
            "X-Title": "Hybrid RAG Pipeline",
        },
    )

    logger.info("OpenRouter LLM ready.")
    return llm


def get_groq_llm(
    model: str = GROQ_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
) -> ChatGroq:
    """Build a ChatGroq fallback LLM instance."""
    if not GROQ_API_KEY:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. Add it to your .env file."
        )

    logger.info(f"Initializing Groq fallback LLM (model='{model}')...")

    llm = ChatGroq(
        model=model,
        temperature=temperature,
        api_key=GROQ_API_KEY,
    )

    logger.info("Groq LLM ready.")
    return llm


# ---------------------------------------------------------------------------
# Smart LLM with auto-fallback
# ---------------------------------------------------------------------------
_primary_llm: Optional[BaseChatModel] = None
_fallback_llm: Optional[BaseChatModel] = None


def get_llm() -> BaseChatModel:
    """
    Return the configured or available LLM.
    Supports LLM_PROVIDER env var ('groq' or 'openrouter', defaulting to 'groq').
    Caches the instance for reuse across calls.
    """
    global _primary_llm
    if _primary_llm is None:
        provider = os.getenv("LLM_PROVIDER", "groq").lower()
        if provider == "groq":
            try:
                _primary_llm = get_groq_llm()
            except Exception as e:
                logger.warning(f"Groq unavailable ({e}), trying OpenRouter...")
                _primary_llm = get_openrouter_llm()
        else:
            try:
                _primary_llm = get_openrouter_llm()
            except Exception as e:
                logger.warning(f"OpenRouter unavailable ({e}), trying Groq...")
                _primary_llm = get_groq_llm()
    return _primary_llm


def _get_fallback_llm() -> BaseChatModel:
    """Return the fallback LLM (Groq). Cached."""
    global _fallback_llm
    if _fallback_llm is None:
        try:
            _fallback_llm = get_groq_llm()
        except EnvironmentError:
            logger.error("Both OpenRouter and Groq are unavailable!")
            raise
    return _fallback_llm


def call_llm_with_fallback(
    chain,
    inputs: dict,
    primary_llm: Optional[BaseChatModel] = None,
) -> str:
    """
    Call a LangChain chain with automatic fallback.

    Tries the primary provider first; if it fails for any reason
    (rate limit, network, bad response), retries with Groq.

    Args:
        chain: A LangChain chain (prompt | llm) or just the prompt template.
        inputs: Dict of inputs for the chain.
        primary_llm: Optional override for the primary LLM.

    Returns:
        The response content string.
    """
    llm = primary_llm or get_llm()

    # If chain is a prompt template, pipe it with the LLM
    if hasattr(chain, "invoke") and not hasattr(chain, "first"):
        # It's already a runnable (prompt | llm), just invoke
        pass
    else:
        chain_with_llm = chain | llm

    try:
        logger.info("Calling primary LLM (OpenRouter)...")
        if hasattr(chain, "first"):
            # It's a RunnableSequence (prompt | llm)
            response = chain.invoke(inputs)
        else:
            response = (chain | llm).invoke(inputs)
        return response.content
    except Exception as e:
        logger.warning(f"Primary LLM failed: {e}. Trying fallback (Groq)...")
        try:
            fallback = _get_fallback_llm()
            if hasattr(chain, "first"):
                # Replace the LLM in the chain
                prompt_part = chain.first
                fallback_chain = prompt_part | fallback
                response = fallback_chain.invoke(inputs)
            else:
                response = (chain | fallback).invoke(inputs)
            return response.content
        except Exception as e2:
            logger.error(f"Fallback LLM also failed: {e2}")
            raise


# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------
def _format_context(documents: List[Document]) -> str:
    """
    Format retrieved Documents into a single context string for the prompt,
    with source attribution per chunk.
    """
    if not documents:
        return "(no relevant context found)"

    formatted_chunks = []
    for i, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "unknown source")
        file_type = doc.metadata.get("file_type", "text")
        label = f"[Chunk {i} | Source: {source} | Type: {file_type}]"
        formatted_chunks.append(f"{label}\n{doc.page_content}")

    return "\n\n".join(formatted_chunks)


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------
def generate_answer(
    question: str,
    retrieved_docs: List[Document],
    llm: Optional[BaseChatModel] = None,
) -> str:
    """
    Generate a grounded answer to `question` using the retrieved chunks
    as context, with automatic primary/fallback provider selection.

    Args:
        question: The user's question.
        retrieved_docs: List of Document chunks from the hybrid retriever.
        llm: Optional pre-built LLM instance. If None, uses the
             cached primary (OpenRouter) with Groq fallback.

    Returns:
        The generated answer string.
    """
    model = llm or get_llm()

    context = _format_context(retrieved_docs)

    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

    logger.info(f"Generating answer for question: '{question}'")

    try:
        chain = prompt | model
        response = chain.invoke({"context": context, "question": question})
        answer = response.content
    except Exception as e:
        logger.warning(f"Primary LLM failed during generation: {e}. Trying fallback...")
        try:
            fallback = _get_fallback_llm()
            chain = prompt | fallback
            response = chain.invoke({"context": context, "question": question})
            answer = response.content
        except Exception as e2:
            logger.error(f"Both LLMs failed: {e2}")
            answer = (
                "I encountered an error generating the answer. "
                "Please check your API keys and try again."
            )

    logger.info("Answer generated.")
    return answer


def stream_answer(
    question: str,
    retrieved_docs: List[Document],
    llm: Optional[BaseChatModel] = None,
):
    """
    Stream tokens for a grounded answer using retrieved chunks as context.
    Yields string chunks progressively. Falls back to secondary provider if primary fails.
    """
    model = llm or get_llm()
    context = _format_context(retrieved_docs)
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

    logger.info(f"Streaming answer for question: '{question}'")
    try:
        chain = prompt | model
        for chunk in chain.stream({"context": context, "question": question}):
            if chunk and hasattr(chunk, "content") and chunk.content:
                yield chunk.content
    except Exception as e:
        logger.warning(f"Primary LLM streaming failed: {e}. Trying fallback...")
        try:
            global _primary_llm
            fallback = _get_fallback_llm()
            _primary_llm = fallback
            fallback_chain = prompt | fallback
            for chunk in fallback_chain.stream({"context": context, "question": question}):
                if chunk and hasattr(chunk, "content") and chunk.content:
                    yield chunk.content
        except Exception as e2:
            logger.error(f"Both LLMs failed during streaming: {e2}")
            yield (
                "I encountered an error generating the answer. "
                "Please check your API keys and try again."
            )


# ---------------------------------------------------------------------------
# Manual test run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from pipeline.data_loader import load_documents
    from pipeline.text_splitter import split_documents
    from pipeline.vectorstore import get_or_build_vectorstore
    from pipeline.search import get_hybrid_retriever, search

    DATA_DIR = os.getenv("DATA_DIR", "./data")

    raw_docs = load_documents(DATA_DIR)
    chunks = split_documents(raw_docs)
    store = get_or_build_vectorstore(chunks)
    retriever = get_hybrid_retriever(chunks, store)

    test_question = "What is this document about?"
    found_docs = search(retriever, test_question)

    final_answer = generate_answer(test_question, found_docs)

    print(f"\nQuestion: {test_question}")
    print(f"\nAnswer:\n{final_answer}")