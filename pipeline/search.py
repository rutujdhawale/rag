"""
search.py

Builds the Hybrid Retriever for the RAG pipeline by combining:
  - FAISS vector retriever  -> semantic understanding
  - BM25 retriever          -> exact keyword matching

Combined via LangChain's EnsembleRetriever, which merges + re-ranks results
from both retrievers using weighted scores.

Usage:
    from data_loader import load_documents
    from text_splitter import split_documents
    from vectorstore import get_or_build_vectorstore
    from search import get_hybrid_retriever

    docs = load_documents("./data")
    chunks = split_documents(docs)
    vectorstore = get_or_build_vectorstore(chunks)

    hybrid_retriever = get_hybrid_retriever(chunks, vectorstore)
    results = hybrid_retriever.invoke("your query here")
"""

import os
import logging
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_community.vectorstores import FAISS

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# Default hybrid weighting: favors semantic slightly over keyword.
# Tune via .env if needed: SEMANTIC_WEIGHT=0.6, KEYWORD_WEIGHT=0.4
DEFAULT_SEMANTIC_WEIGHT = float(os.getenv("SEMANTIC_WEIGHT", 0.6))
DEFAULT_KEYWORD_WEIGHT = float(os.getenv("KEYWORD_WEIGHT", 0.4))
DEFAULT_TOP_K = int(os.getenv("RETRIEVER_TOP_K", 4))


def build_bm25_retriever(
    chunks: List[Document],
    top_k: int = DEFAULT_TOP_K,
) -> BM25Retriever:
    """
    Build a BM25 retriever (exact keyword matching) from chunked Documents.

    Note: BM25Retriever is in-memory and rebuilt from the chunk list each
    run — there's no "save to disk" step like FAISS. Keep your chunks
    available (or reload + re-split) whenever you need to rebuild this.

    Args:
        chunks: List of chunked Document objects.
        top_k: Number of results to return per query.

    Returns:
        A BM25Retriever instance.
    """
    if not chunks:
        raise ValueError("Cannot build BM25 retriever: no chunks provided.")

    logger.info(f"Building BM25 keyword retriever from {len(chunks)} chunk(s)...")
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = top_k

    logger.info("BM25 retriever built successfully.")
    return bm25_retriever


def build_vector_retriever(
    vectorstore: FAISS,
    top_k: int = DEFAULT_TOP_K,
):
    """
    Build a semantic retriever from an existing FAISS vectorstore.

    Args:
        vectorstore: A FAISS vectorstore instance (from vectorstore.py).
        top_k: Number of results to return per query.

    Returns:
        A LangChain retriever (VectorStoreRetriever).
    """
    logger.info("Building FAISS semantic retriever...")
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    logger.info("FAISS retriever built successfully.")
    return vector_retriever


def get_hybrid_retriever(
    chunks: List[Document],
    vectorstore: FAISS,
    semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
    keyword_weight: float = DEFAULT_KEYWORD_WEIGHT,
    top_k: int = DEFAULT_TOP_K,
) -> EnsembleRetriever:
    """
    Build the combined Hybrid Retriever (FAISS semantic + BM25 keyword).

    Args:
        chunks: List of chunked Document objects (needed to build BM25).
        vectorstore: A FAISS vectorstore instance (needed for semantic search).
        semantic_weight: Weight given to vector/semantic results (0-1).
        keyword_weight: Weight given to BM25/keyword results (0-1).
            semantic_weight + keyword_weight should sum to 1.0.
        top_k: Number of results each individual retriever returns
            before merging/re-ranking.

    Returns:
        An EnsembleRetriever combining both retrieval strategies.
    """
    if round(semantic_weight + keyword_weight, 5) != 1.0:
        logger.warning(
            f"Weights don't sum to 1.0 (got {semantic_weight} + {keyword_weight} "
            f"= {semantic_weight + keyword_weight}). EnsembleRetriever will still "
            f"run, but consider normalizing them."
        )

    bm25_retriever = build_bm25_retriever(chunks, top_k=top_k)
    vector_retriever = build_vector_retriever(vectorstore, top_k=top_k)

    logger.info(
        f"Combining retrievers via EnsembleRetriever "
        f"(semantic={semantic_weight}, keyword={keyword_weight})..."
    )

    hybrid_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[semantic_weight, keyword_weight],
    )

    logger.info("Hybrid retriever ready.")
    return hybrid_retriever


def search(
    hybrid_retriever: EnsembleRetriever,
    query: str,
) -> List[Document]:
    """
    Run a query through the hybrid retriever.

    Args:
        hybrid_retriever: An EnsembleRetriever instance from get_hybrid_retriever().
        query: The search query string.

    Returns:
        List of relevant Document chunks, merged + re-ranked from both
        semantic and keyword retrieval.
    """
    logger.info(f"Running hybrid search for query: '{query}'")
    results = hybrid_retriever.invoke(query)
    logger.info(f"Hybrid search returned {len(results)} result(s).")
    return results


def detailed_hybrid_retrieve(
    query: str,
    vectorstore: FAISS,
    bm25_retriever: BM25Retriever,
    semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
    keyword_weight: float = DEFAULT_KEYWORD_WEIGHT,
    top_k: int = DEFAULT_TOP_K,
    rrf_k: int = 60,
) -> tuple[List[Document], dict]:
    """
    Perform hybrid retrieval with explicit telemetry for transparency.
    Executes dense vector search and sparse BM25 search, computes weighted
    Reciprocal Rank Fusion (RRF), deduplicates chunks, and returns both
    the final selected documents and detailed inspection telemetry.

    Returns:
        (final_docs, telemetry_dict)
    """
    logger.info(f"Running detailed hybrid search: query='{query}', top_k={top_k}")

    # 1. Dense retrieval (Vector)
    vector_results = []
    try:
        # Try similarity_search_with_score (FAISS L2 distance: lower is closer)
        docs_and_scores = vectorstore.similarity_search_with_score(query, k=top_k)
        for rank, (doc, dist) in enumerate(docs_and_scores, start=1):
            # Convert L2 distance to an approximate 0-1 similarity score
            sim_score = round(1.0 / (1.0 + float(dist)), 4)
            doc_copy = Document(page_content=doc.page_content, metadata={**doc.metadata})
            doc_copy.metadata["retrieval_method"] = "vector"
            doc_copy.metadata["vector_rank"] = rank
            doc_copy.metadata["vector_score"] = sim_score
            vector_results.append(doc_copy)
    except Exception as e:
        logger.warning(f"Vector search with score failed ({e}), falling back to standard search.")
        raw_docs = vectorstore.similarity_search(query, k=top_k)
        for rank, doc in enumerate(raw_docs, start=1):
            doc_copy = Document(page_content=doc.page_content, metadata={**doc.metadata})
            doc_copy.metadata["retrieval_method"] = "vector"
            doc_copy.metadata["vector_rank"] = rank
            vector_results.append(doc_copy)

    # 2. Sparse retrieval (BM25)
    bm25_results = []
    try:
        raw_bm25 = bm25_retriever.invoke(query)
        for rank, doc in enumerate(raw_bm25[:top_k], start=1):
            doc_copy = Document(page_content=doc.page_content, metadata={**doc.metadata})
            doc_copy.metadata["retrieval_method"] = "bm25"
            doc_copy.metadata["bm25_rank"] = rank
            bm25_results.append(doc_copy)
    except Exception as e:
        logger.warning(f"BM25 search failed: {e}")

    # 3. Weighted Reciprocal Rank Fusion (RRF)
    # Deduplicate by (source, chunk_index) or trimmed page_content
    fused_scores: dict[str, float] = {}
    doc_registry: dict[str, Document] = {}
    membership: dict[str, dict] = {}

    def get_doc_key(doc: Document) -> str:
        source = doc.metadata.get("source", "")
        chunk_idx = doc.metadata.get("chunk_index")
        if source and chunk_idx is not None:
            return f"{source}::{chunk_idx}"
        return doc.page_content[:120].strip()

    # Process vector hits
    for rank, doc in enumerate(vector_results, start=1):
        key = get_doc_key(doc)
        rrf_contrib = semantic_weight / (rrf_k + rank)
        fused_scores[key] = fused_scores.get(key, 0.0) + rrf_contrib
        if key not in doc_registry:
            doc_registry[key] = doc
        membership[key] = membership.get(key, {})
        membership[key]["in_vector"] = True
        membership[key]["vector_rank"] = rank
        membership[key]["vector_score"] = doc.metadata.get("vector_score")

    # Process BM25 hits
    for rank, doc in enumerate(bm25_results, start=1):
        key = get_doc_key(doc)
        rrf_contrib = keyword_weight / (rrf_k + rank)
        fused_scores[key] = fused_scores.get(key, 0.0) + rrf_contrib
        if key not in doc_registry:
            doc_registry[key] = doc
        membership[key] = membership.get(key, {})
        membership[key]["in_bm25"] = True
        membership[key]["bm25_rank"] = rank

    # Sort merged results by fused score desc
    sorted_keys = sorted(fused_scores.keys(), key=lambda k: fused_scores[k], reverse=True)

    final_docs: List[Document] = []
    fused_telemetry: list[dict] = []

    for key in sorted_keys:
        base_doc = doc_registry[key]
        m = membership[key]
        score = round(fused_scores[key], 6)

        # Annotate doc metadata with hybrid provenance
        merged_doc = Document(
            page_content=base_doc.page_content,
            metadata={
                **base_doc.metadata,
                "rrf_score": score,
                "in_vector": m.get("in_vector", False),
                "in_bm25": m.get("in_bm25", False),
                "vector_rank": m.get("vector_rank"),
                "bm25_rank": m.get("bm25_rank"),
                "vector_score": m.get("vector_score"),
                "retrieval_method": (
                    "hybrid_both" if (m.get("in_vector") and m.get("in_bm25"))
                    else "vector_only" if m.get("in_vector")
                    else "bm25_only"
                ),
            },
        )
        if len(final_docs) < top_k:
            final_docs.append(merged_doc)

        fused_telemetry.append({
            "key": key,
            "source": Path(base_doc.metadata.get("source", "unknown")).name,
            "chunk_index": base_doc.metadata.get("chunk_index", 0),
            "file_type": base_doc.metadata.get("file_type", "unknown"),
            "rrf_score": score,
            "in_vector": m.get("in_vector", False),
            "in_bm25": m.get("in_bm25", False),
            "vector_rank": m.get("vector_rank"),
            "bm25_rank": m.get("bm25_rank"),
            "preview": base_doc.page_content[:200],
        })

    telemetry = {
        "strategy": "Hybrid Dense (FAISS) + Sparse (BM25)",
        "fusion_algorithm": f"Weighted Reciprocal Rank Fusion (k={rrf_k})",
        "semantic_weight": semantic_weight,
        "keyword_weight": keyword_weight,
        "top_k": top_k,
        "vector_candidate_count": len(vector_results),
        "bm25_candidate_count": len(bm25_results),
        "fused_candidate_count": len(sorted_keys),
        "final_context_count": len(final_docs),
        "overlap_count": sum(1 for m in membership.values() if m.get("in_vector") and m.get("in_bm25")),
        "vector_hits": [
            {
                "source": Path(d.metadata.get("source", "unknown")).name,
                "chunk_index": d.metadata.get("chunk_index", 0),
                "rank": d.metadata.get("vector_rank"),
                "score": d.metadata.get("vector_score"),
                "preview": d.page_content[:180],
            }
            for d in vector_results
        ],
        "bm25_hits": [
            {
                "source": Path(d.metadata.get("source", "unknown")).name,
                "chunk_index": d.metadata.get("chunk_index", 0),
                "rank": d.metadata.get("bm25_rank"),
                "preview": d.page_content[:180],
            }
            for d in bm25_results
        ],
        "fused_hits": fused_telemetry,
    }

    return final_docs, telemetry


# ---------------------------------------------------------------------------
# Manual test run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from pipeline.data_loader import load_documents
    from pipeline.text_splitter import split_documents
    from pipeline.vectorstore import get_or_build_vectorstore

    DATA_DIR = os.getenv("DATA_DIR", "./data")

    raw_docs = load_documents(DATA_DIR)
    doc_chunks = split_documents(raw_docs)
    store = get_or_build_vectorstore(doc_chunks)

    retriever = get_hybrid_retriever(doc_chunks, store)

    test_query = "test query"
    found_docs = search(retriever, test_query)

    print(f"\nTop {len(found_docs)} result(s) for query: '{test_query}'")
    for i, doc in enumerate(found_docs):
        print(f"\n--- Result {i+1} ---")
        print(f"Source: {doc.metadata.get('source')}")
        print(f"Type: {doc.metadata.get('file_type')}")
        print(f"Content preview: {doc.page_content[:200]}")