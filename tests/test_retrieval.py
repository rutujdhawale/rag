"""
tests/test_retrieval.py

Unit and integration tests for retrieval components:
- BM25 sparse keyword retrieval
- Vector dense semantic retrieval
- Hybrid RRF score fusion
- Deduplication
- Top-k candidate limits
"""

import pytest
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings

from pipeline.search import (
    build_bm25_retriever,
    build_vector_retriever,
    get_hybrid_retriever,
    detailed_hybrid_retrieve,
)


@pytest.fixture
def sample_documents():
    return [
        Document(
            page_content="Phase Shift Keying (PSK) is a digital modulation technique.",
            metadata={"source": "doc1.pdf", "chunk_index": 0, "file_type": "pdf"},
        ),
        Document(
            page_content="Orthogonal Frequency Division Multiplexing divides available spectrum.",
            metadata={"source": "doc1.pdf", "chunk_index": 1, "file_type": "pdf"},
        ),
        Document(
            page_content="Hospital bed allocation and patient records management database system.",
            metadata={"source": "doc2.pdf", "chunk_index": 0, "file_type": "pdf"},
        ),
        Document(
            page_content="Transmission Control Protocol ensures reliable delivery of data packets.",
            metadata={"source": "doc3.pdf", "chunk_index": 0, "file_type": "pdf"},
        ),
    ]


@pytest.fixture
def fake_vectorstore(sample_documents):
    # FakeEmbeddings with size 384
    embeddings = FakeEmbeddings(size=384)
    return FAISS.from_documents(sample_documents, embeddings)


def test_bm25_exact_keyword_matching(sample_documents):
    bm25 = build_bm25_retriever(sample_documents, top_k=2)
    results = bm25.invoke("PSK modulation")
    assert len(results) > 0
    assert "PSK" in results[0].page_content
    assert results[0].metadata["source"] == "doc1.pdf"


def test_bm25_empty_chunks_raises_error():
    with pytest.raises(ValueError):
        build_bm25_retriever([], top_k=3)


def test_vector_retrieval(fake_vectorstore):
    retriever = build_vector_retriever(fake_vectorstore, top_k=2)
    results = retriever.invoke("digital modulation")
    assert len(results) <= 2
    assert all(isinstance(doc, Document) for doc in results)


def test_hybrid_detailed_fusion_and_deduplication(sample_documents, fake_vectorstore):
    bm25 = build_bm25_retriever(sample_documents, top_k=3)
    results, telemetry = detailed_hybrid_retrieve(
        query="Phase Shift Keying",
        vectorstore=fake_vectorstore,
        bm25_retriever=bm25,
        semantic_weight=0.6,
        keyword_weight=0.4,
        top_k=2,
    )

    assert len(results) <= 2
    assert "vector_candidate_count" in telemetry
    assert "bm25_candidate_count" in telemetry
    assert "fused_candidate_count" in telemetry
    assert "fused_hits" in telemetry
    assert telemetry["top_k"] == 2

    # Check that chunks are deduplicated (no duplicate source + chunk_index)
    keys = [f"{d.metadata.get('source')}::{d.metadata.get('chunk_index')}" for d in results]
    assert len(keys) == len(set(keys))


def test_hybrid_retriever_custom_weights(sample_documents, fake_vectorstore):
    hybrid = get_hybrid_retriever(
        sample_documents,
        fake_vectorstore,
        semantic_weight=0.8,
        keyword_weight=0.2,
        top_k=3,
    )
    assert hybrid is not None
    results = hybrid.invoke("Transmission Control Protocol")
    assert len(results) > 0
