"""
core/engine.py

Orchestration layer bridging storage and the RAG pipeline.
Implements in-memory caching for models and vector indices to prevent repeated
initializations across pipeline executions.
"""

import functools
import logging
import os
import time
from pathlib import Path
from typing import Callable, Generator, List, Optional, Tuple

from langchain_core.documents import Document

from pipeline.data_loader import load_documents
from pipeline.text_splitter import split_documents
from pipeline.embedding import get_embedding_model
from pipeline.vectorstore import (
    load_vectorstore,
    build_vectorstore,
    get_or_build_vectorstore,
)
from pipeline.search import (
    build_bm25_retriever,
    get_hybrid_retriever,
    detailed_hybrid_retrieve,
)
from pipeline.llm import (
    get_openrouter_llm,
    get_groq_llm,
    get_llm,
    generate_answer,
    stream_answer,
)
from pipeline.graph import build_graph, run_query
from storage.doc_store import record_document_indexed, remove_document

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# In-Memory Caching for Heavy Resources
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=4)
def get_cached_embedding(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Cache embedding model once per process lifetime."""
    logger.info(f"Loading cached embedding model: {model_name}")
    return get_embedding_model(model_name=model_name)


@functools.lru_cache(maxsize=8)
def get_cached_llm(provider: str, model_name: str, temperature: float):
    """Cache LLM instance for a given provider, model, and temperature."""
    logger.info(f"Initializing cached LLM: provider={provider}, model={model_name}")
    if provider == "groq":
        return get_groq_llm(model=model_name, temperature=temperature)
    return get_openrouter_llm(model=model_name, temperature=temperature)


# ---------------------------------------------------------------------------
# Document Ingestion & Indexing Pipeline
# ---------------------------------------------------------------------------
def save_uploaded_files(uploaded_files, data_dir: str = "./data") -> list[Path]:
    """Save user-uploaded files to data_dir."""
    target_dir = Path(data_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    for uf in uploaded_files:
        dest = target_dir / uf.name
        with open(dest, "wb") as f:
            f.write(uf.getbuffer())
        saved_paths.append(dest)
        logger.info(f"Saved uploaded file: {dest}")

    return saved_paths


def run_indexing_pipeline(
    data_dir: str = "./data",
    index_dir: str = "./faiss_index",
    settings: Optional[dict] = None,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> Tuple[Optional[object], Optional[object], List[Document], dict]:
    """
    Executes the ingestion stages with real progress tracking:
      1. Validate documents
      2. Extract text & image descriptions
      3. Create chunks with overlap
      4. Generate embeddings & build FAISS index
      5. Build in-memory BM25 index
      6. Update document catalog
    """
    cfg = settings or {}
    chunk_size = cfg.get("chunk_size", 1000)
    chunk_overlap = cfg.get("chunk_overlap", 150)
    top_k = cfg.get("top_k", 4)
    model_name = cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")

    def notify(msg: str, pct: float):
        if progress_callback:
            progress_callback(msg, pct)
        logger.info(f"Indexing [{int(pct*100)}%]: {msg}")

    # Stage 1: Validate files in data_dir
    notify("Validating document directory...", 0.1)
    dir_path = Path(data_dir)
    if not dir_path.exists() or not any(dir_path.iterdir()):
        raise ValueError(f"No documents found in '{data_dir}'. Please upload files first.")

    # Stage 2: Load raw documents
    notify("Extracting text and multimodal content...", 0.25)
    t0 = time.time()
    raw_docs = load_documents(data_dir)
    load_time = time.time() - t0

    if not raw_docs:
        raise ValueError(f"No extractable text or content found in '{data_dir}'.")

    # Stage 3: Split into chunks
    notify(f"Splitting {len(raw_docs)} document sections into chunks...", 0.45)
    chunks = split_documents(raw_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        raise ValueError("Failed to create document chunks.")

    # Stage 4: Embeddings & FAISS
    notify(f"Generating embeddings & updating FAISS index for {len(chunks)} chunks...", 0.65)
    embedding_model = get_cached_embedding(model_name)
    vectorstore = build_vectorstore(chunks, index_dir=index_dir, embedding_model=embedding_model)

    # Stage 5: In-memory BM25 index
    notify("Building BM25 keyword index...", 0.85)
    bm25 = build_bm25_retriever(chunks, top_k=top_k)

    # Stage 6: Update persistent catalog
    notify("Recording indexed document catalog...", 0.95)
    # Count chunks per source file
    chunks_per_file: dict[str, int] = {}
    for c in chunks:
        src = Path(c.metadata.get("source", "")).name
        chunks_per_file[src] = chunks_per_file.get(src, 0) + 1

    for file_path in dir_path.iterdir():
        if file_path.is_file():
            name = file_path.name
            record_document_indexed(
                filename=name,
                file_type=file_path.suffix.lstrip("."),
                size_bytes=file_path.stat().st_size,
                chunk_count=chunks_per_file.get(name, 0),
            )

    # Save chunks cache to disk for instant loading on subsequent runs
    try:
        import pickle
        chunks_cache_file = Path(index_dir) / "chunks.pkl"
        with open(chunks_cache_file, "wb") as f:
            pickle.dump(chunks, f)
        logger.info(f"Saved {len(chunks)} chunks to '{chunks_cache_file}'")
    except Exception as e:
        logger.warning(f"Could not cache chunks: {e}")

    notify("Indexing complete. Pipeline is ready.", 1.0)

    stats = {
        "raw_doc_count": len(raw_docs),
        "chunk_count": len(chunks),
        "load_time_s": round(load_time, 2),
    }

    return vectorstore, bm25, chunks, stats


def load_existing_pipeline(
    data_dir: str = "./data",
    index_dir: str = "./faiss_index",
    settings: Optional[dict] = None,
) -> Tuple[Optional[object], Optional[object], List[Document]]:
    """
    Load an already indexed FAISS index from disk and build BM25 from cached chunks.
    Skips embedding regeneration and expensive document re-parsing for instant startup!
    """
    cfg = settings or {}
    model_name = cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
    top_k = cfg.get("top_k", 4)
    chunk_size = cfg.get("chunk_size", 1000)
    chunk_overlap = cfg.get("chunk_overlap", 150)

    faiss_file = Path(index_dir) / "index.faiss"
    if not faiss_file.exists():
        return None, None, []

    embedding_model = get_cached_embedding(model_name)
    vectorstore = load_vectorstore(index_dir, embedding_model=embedding_model)

    # 1. Try loading cached chunks for instant BM25 generation
    chunks_cache_file = Path(index_dir) / "chunks.pkl"
    chunks = []
    if chunks_cache_file.exists():
        try:
            import pickle
            with open(chunks_cache_file, "rb") as f:
                chunks = pickle.load(f)
            logger.info(f"Loaded {len(chunks)} cached chunks from '{chunks_cache_file}'")
        except Exception as e:
            logger.warning(f"Failed loading cached chunks: {e}")

    # 2. If no chunk cache exists, load and split from disk once, then cache
    if not chunks:
        raw_docs = load_documents(data_dir)
        chunks = split_documents(raw_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        try:
            import pickle
            with open(chunks_cache_file, "wb") as f:
                pickle.dump(chunks, f)
        except Exception:
            pass

    bm25 = build_bm25_retriever(chunks, top_k=top_k) if chunks else None

    return vectorstore, bm25, chunks


# ---------------------------------------------------------------------------
# Query Execution & Hybrid Search Orchestration
# ---------------------------------------------------------------------------
def run_hybrid_query(
    question: str,
    vectorstore,
    bm25_retriever,
    settings: Optional[dict] = None,
) -> dict:
    """
    Executes hybrid retrieval, score fusion, and answers generation.
    Returns answer, sources, chunk telemetry, and execution timing.
    """
    cfg = settings or {}
    semantic_weight = cfg.get("hybrid_weight", 0.6)
    keyword_weight = round(1.0 - semantic_weight, 2)
    top_k = cfg.get("top_k", 4)
    provider = cfg.get("llm_provider", "openrouter")
    temperature = cfg.get("temperature", 0.2)
    model_name = (
        cfg.get("openrouter_model", "google/gemma-3-27b-it:free")
        if provider == "openrouter"
        else cfg.get("groq_model", "llama-3.3-70b-versatile")
    )

    t0 = time.time()

    # 1. Detailed Hybrid Retrieval
    t_ret_start = time.time()
    final_docs, telemetry = detailed_hybrid_retrieve(
        query=question,
        vectorstore=vectorstore,
        bm25_retriever=bm25_retriever,
        semantic_weight=semantic_weight,
        keyword_weight=keyword_weight,
        top_k=top_k,
    )
    retrieval_time = round(time.time() - t_ret_start, 3)

    # 2. Get cached LLM
    try:
        llm = get_cached_llm(provider, model_name, temperature)
    except Exception as e:
        logger.warning(f"Error initializing configured LLM ({e}), using default fallback.")
        llm = get_llm()

    # 3. Generate Answer
    t_gen_start = time.time()
    answer = generate_answer(question, final_docs, llm=llm)
    generation_time = round(time.time() - t_gen_start, 3)
    total_time = round(time.time() - t0, 3)

    # 4. Extract clean, unique source summaries
    sources_summary = []
    seen_sources = set()
    for doc in final_docs:
        src = Path(doc.metadata.get("source", "Unknown")).name
        page = doc.metadata.get("page", doc.metadata.get("page_number"))
        chunk_idx = doc.metadata.get("chunk_index", 0)
        ret_method = doc.metadata.get("retrieval_method", "hybrid")
        rrf = doc.metadata.get("rrf_score", 0.0)

        key = f"{src}::{page}::{chunk_idx}"
        if key not in seen_sources:
            seen_sources.add(key)
            sources_summary.append({
                "filename": src,
                "page": page,
                "chunk_index": chunk_idx,
                "retrieval_method": ret_method,
                "score": rrf,
                "preview": doc.page_content[:300].strip(),
            })

    telemetry["retrieval_time_s"] = retrieval_time
    telemetry["generation_time_s"] = generation_time
    telemetry["total_time_s"] = total_time
    telemetry["provider"] = provider
    telemetry["model"] = model_name

    return {
        "answer": answer,
        "sources": sources_summary,
        "retrieval_details": telemetry,
        "timing_s": total_time,
        "documents": final_docs,
    }


def stream_hybrid_query(
    question: str,
    vectorstore,
    bm25_retriever,
    settings: Optional[dict] = None,
) -> Tuple[Generator[str, None, None], dict]:
    """
    Executes hybrid retrieval first, prepares telemetry and source metadata,
    and returns (token_generator, metadata_dict) for smooth UI streaming.
    """
    cfg = settings or {}
    semantic_weight = cfg.get("hybrid_weight", 0.6)
    keyword_weight = round(1.0 - semantic_weight, 2)
    top_k = cfg.get("top_k", 4)
    provider = cfg.get("llm_provider", "openrouter")
    temperature = cfg.get("temperature", 0.2)
    model_name = (
        cfg.get("openrouter_model", "google/gemma-3-27b-it:free")
        if provider == "openrouter"
        else cfg.get("groq_model", "llama-3.3-70b-versatile")
    )

    t0 = time.time()

    # Detailed hybrid retrieval
    final_docs, telemetry = detailed_hybrid_retrieve(
        query=question,
        vectorstore=vectorstore,
        bm25_retriever=bm25_retriever,
        semantic_weight=semantic_weight,
        keyword_weight=keyword_weight,
        top_k=top_k,
    )
    retrieval_time = round(time.time() - t0, 3)

    # Get LLM
    try:
        llm = get_cached_llm(provider, model_name, temperature)
    except Exception as e:
        logger.warning(f"Error initializing configured LLM ({e}), using default.")
        llm = get_llm()

    # Build sources summary
    sources_summary = []
    seen = set()
    for doc in final_docs:
        src = Path(doc.metadata.get("source", "Unknown")).name
        page = doc.metadata.get("page", doc.metadata.get("page_number"))
        chunk_idx = doc.metadata.get("chunk_index", 0)
        ret_method = doc.metadata.get("retrieval_method", "hybrid")
        rrf = doc.metadata.get("rrf_score", 0.0)

        key = f"{src}::{page}::{chunk_idx}"
        if key not in seen:
            seen.add(key)
            sources_summary.append({
                "filename": src,
                "page": page,
                "chunk_index": chunk_idx,
                "retrieval_method": ret_method,
                "score": rrf,
                "preview": doc.page_content[:300].strip(),
            })

    telemetry["retrieval_time_s"] = retrieval_time
    telemetry["provider"] = provider
    telemetry["model"] = model_name

    meta = {
        "sources": sources_summary,
        "retrieval_details": telemetry,
        "retrieval_time_s": retrieval_time,
        "documents": final_docs,
    }

    # Token streaming generator
    token_generator = stream_answer(question, final_docs, llm=llm)

    return token_generator, meta
