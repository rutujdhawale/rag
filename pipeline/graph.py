"""
graph.py

LangGraph flow for the Hybrid RAG pipeline:

    retrieve -> grade_relevance -> [relevant?] -> generate
                                  -> [not relevant?] -> rewrite_query -> retrieve (retry)
                                  -> [max retries hit] -> generate (fallback / "I don't know")

This makes the pipeline self-correcting: if the hybrid retriever pulls back
chunks that don't actually answer the question, the graph rewrites the query
(e.g. fixing phrasing, broadening terms) and tries retrieval again, instead
of blindly generating an answer from irrelevant context.

Usage:
    from data_loader import load_documents
    from text_splitter import split_documents
    from vectorstore import get_or_build_vectorstore
    from search import get_hybrid_retriever
    from graph import build_graph

    docs = load_documents("./data")
    chunks = split_documents(docs)
    store = get_or_build_vectorstore(chunks)
    retriever = get_hybrid_retriever(chunks, store)

    app = build_graph(retriever)
    result = app.invoke({"question": "your question here"})
    print(result["answer"])
"""

import os
import sys
import logging
from typing import List, TypedDict, Optional

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

from pipeline.llm import get_llm, generate_answer, call_llm_with_fallback

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MAX_RETRIES = int(os.getenv("MAX_RETRIEVAL_RETRIES", 2))

FALLBACK_ANSWER = (
    "I couldn't find enough relevant information in the provided documents "
    "to answer this question confidently. Please try rephrasing your question "
    "or check that relevant documents have been added to the knowledge base."
)


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------
class GraphState(TypedDict):
    question: str               # current (possibly rewritten) query
    original_question: str      # the user's original question, kept for final generation
    documents: List[Document]   # retrieved chunks
    is_relevant: bool           # grading result
    retry_count: int            # how many retrieval retries have happened
    answer: str                 # final generated answer


# ---------------------------------------------------------------------------
# Grading prompt
# ---------------------------------------------------------------------------
GRADE_PROMPT_TEMPLATE = """You are a grader assessing whether retrieved document chunks are relevant to a user's question.

Question: {question}

Retrieved chunks:
{context}

Are these chunks relevant enough to answer the question, at least partially?
If the chunks contain keywords, concepts, summaries, or topics related to the question, respond "yes".
Only respond "no" if the chunks are completely and totally off-topic.
Respond with ONLY one word: "yes" or "no". No explanation, no punctuation."""

REWRITE_PROMPT_TEMPLATE = """The following search query did not retrieve relevant results from a document knowledge base.

Original query: {question}

Rewrite this query to be clearer, broader, or use different phrasing/synonyms that might
match the documents better. Respond with ONLY the rewritten query, nothing else."""


def _format_context(documents: List[Document]) -> str:
    if not documents:
        return "(no chunks retrieved)"
    return "\n\n".join(
        f"[Chunk {i+1}] {doc.page_content[:1000]}" for i, doc in enumerate(documents)
    )


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------
def make_retrieve_node(hybrid_retriever):
    """Closure so the retrieve node has access to the hybrid_retriever instance."""

    def retrieve(state: GraphState) -> GraphState:
        logger.info(f"[retrieve] Query: '{state['question']}'")
        docs = hybrid_retriever.invoke(state["question"])
        logger.info(f"[retrieve] Got {len(docs)} chunk(s).")
        # Keep previous documents if new retrieval returns empty
        existing = state.get("documents", [])
        if not docs and existing:
            docs = existing
        return {**state, "documents": docs}

    return retrieve


def make_grade_node():
    """Closure so the grade node assesses relevance with fallback protection."""
    prompt = ChatPromptTemplate.from_template(GRADE_PROMPT_TEMPLATE)

    def grade_relevance(state: GraphState) -> GraphState:
        context = _format_context(state["documents"])

        if context == "(no chunks retrieved)":
            logger.info("[grade] No chunks retrieved — marking as not relevant.")
            return {**state, "is_relevant": False}

        logger.info("[grade] Grading retrieved chunks for relevance...")
        try:
            content = call_llm_with_fallback(prompt, {"question": state["question"], "context": context})
            verdict = str(content).strip().lower()
        except Exception as e:
            logger.warning(f"[grade] Grading LLM call failed ({e}) — defaulting to relevant.")
            verdict = "yes"

        # Lenient relevance check: only mark not relevant if explicitly 'no' or 'not relevant'
        verdict_clean = verdict.strip().lower()
        if verdict_clean in ("no", "no.", "not relevant") or verdict_clean.startswith("no\n"):
            is_relevant = False
        else:
            is_relevant = True

        logger.info(f"[grade] Verdict: {verdict} -> is_relevant={is_relevant}")

        return {**state, "is_relevant": is_relevant}

    return grade_relevance


def make_rewrite_node():
    """Closure so the rewrite node rewrites query with fallback protection."""
    prompt = ChatPromptTemplate.from_template(REWRITE_PROMPT_TEMPLATE)

    def rewrite_query(state: GraphState) -> GraphState:
        logger.info(f"[rewrite] Rewriting query: '{state['question']}'")
        try:
            content = call_llm_with_fallback(prompt, {"question": state["question"]})
            new_question = str(content).strip().strip('"').strip("'")
            # Strip common conversational LLM prefixes
            for prefix in ("rewritten query:", "search query:", "query:", "here is the rewritten query:"):
                if new_question.lower().startswith(prefix):
                    new_question = new_question[len(prefix):].strip().strip('"').strip("'")
        except Exception as e:
            logger.warning(f"[rewrite] Query rewrite failed ({e}) — keeping current query.")
            new_question = state["question"]

        logger.info(f"[rewrite] Rewritten query: '{new_question}'")

        return {
            **state,
            "question": new_question,
            "retry_count": state["retry_count"] + 1,
        }

    return rewrite_query


def make_generate_node():
    """Closure for final generation node with automatic fallback."""
    def generate(state: GraphState) -> GraphState:
        if not state.get("documents"):
            logger.info("[generate] No documents found — using fallback answer.")
            return {**state, "answer": FALLBACK_ANSWER}

        logger.info("[generate] Generating answer from retrieved context...")
        answer = generate_answer(
            question=state["original_question"],
            retrieved_docs=state["documents"],
        )
        return {**state, "answer": answer}

    return generate


# ---------------------------------------------------------------------------
# Conditional routing
# ---------------------------------------------------------------------------
def route_after_grading(state: GraphState) -> str:
    """
    Decide what to do after grading:
      - relevant -> go straight to generate
      - not relevant, retries left -> rewrite query and retry retrieval
      - not relevant, no retries left -> generate fallback answer
    """
    if state["is_relevant"]:
        return "generate"

    if state["retry_count"] < MAX_RETRIES:
        return "rewrite_query"

    logger.info(f"[route] Max retries ({MAX_RETRIES}) reached — falling back.")
    return "generate"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------
def build_graph(hybrid_retriever):
    """
    Build and compile the LangGraph RAG flow.

    Args:
        hybrid_retriever: An EnsembleRetriever instance from search.get_hybrid_retriever().

    Returns:
        A compiled LangGraph app. Call .invoke({"question": "..."}) to run it.
    """
    graph = StateGraph(GraphState)

    graph.add_node("retrieve", make_retrieve_node(hybrid_retriever))
    graph.add_node("grade_relevance", make_grade_node())
    graph.add_node("rewrite_query", make_rewrite_node())
    graph.add_node("generate", make_generate_node())

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade_relevance")

    graph.add_conditional_edges(
        "grade_relevance",
        route_after_grading,
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query",
        },
    )

    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("generate", END)

    app = graph.compile()
    logger.info("LangGraph RAG flow compiled successfully.")

    return app


def run_query(app, question: str) -> dict:
    """
    Run a question through the compiled graph and return the full result
    including answer, retrieved documents, relevance grading, and timing.

    Args:
        app: Compiled graph from build_graph().
        question: The user's question.

    Returns:
        Dict with keys: answer, documents, is_relevant, retry_count,
        question (possibly rewritten), original_question, timing_s.
    """
    import time

    initial_state: GraphState = {
        "question": question,
        "original_question": question,
        "documents": [],
        "is_relevant": False,
        "retry_count": 0,
        "answer": "",
    }

    t_start = time.time()
    result = app.invoke(initial_state)
    t_total = time.time() - t_start

    return {
        "answer": result.get("answer", ""),
        "documents": result.get("documents", []),
        "is_relevant": result.get("is_relevant", False),
        "retry_count": result.get("retry_count", 0),
        "question": result.get("question", question),
        "original_question": question,
        "timing_s": round(t_total, 3),
    }


# ---------------------------------------------------------------------------
# Manual test run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from pipeline.data_loader import load_documents
    from pipeline.text_splitter import split_documents
    from pipeline.vectorstore import get_or_build_vectorstore
    from pipeline.search import get_hybrid_retriever

    DATA_DIR = os.getenv("DATA_DIR", "./data")

    raw_docs = load_documents(DATA_DIR)
    chunks = split_documents(raw_docs)
    store = get_or_build_vectorstore(chunks)
    retriever = get_hybrid_retriever(chunks, store)

    rag_app = build_graph(retriever)

    test_question = "What is this document about?"
    final_answer = run_query(rag_app, test_question)

    print(f"\nQuestion: {test_question}")
    print(f"\nAnswer:\n{final_answer}")