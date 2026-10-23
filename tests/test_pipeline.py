"""
tests/test_pipeline.py

Unit tests for pipeline utilities:
- Text splitter boundaries and metadata preservation
- Context formatting for RAG prompt
- LangGraph routing and maximum retry conditions
"""

from langchain_core.documents import Document

from pipeline.text_splitter import split_documents
from pipeline.llm import _format_context
from pipeline.graph import route_after_grading, GraphState, MAX_RETRIES


def test_text_splitter_preserves_metadata():
    docs = [
        Document(
            page_content="A " * 600,  # 1200 characters
            metadata={"source": "test.txt", "file_type": "txt"},
        )
    ]
    chunks = split_documents(docs, chunk_size=500, chunk_overlap=50)
    assert len(chunks) > 1
    for idx, c in enumerate(chunks):
        assert c.metadata["source"] == "test.txt"
        assert c.metadata["file_type"] == "txt"
        assert c.metadata["chunk_index"] == idx


def test_text_splitter_skips_atomic_csv():
    docs = [
        Document(
            page_content="col1,col2\nval1,val2",
            metadata={"source": "data.csv", "file_type": "csv"},
        )
    ]
    chunks = split_documents(docs, chunk_size=500, chunk_overlap=50)
    assert len(chunks) == 1
    assert chunks[0].page_content == docs[0].page_content


def test_context_formatting_attribution():
    chunks = [
        Document(page_content="First piece of evidence.", metadata={"source": "docA.pdf", "file_type": "pdf"}),
        Document(page_content="Second piece of evidence.", metadata={"source": "docB.docx", "file_type": "docx"}),
    ]
    formatted = _format_context(chunks)
    assert "[Chunk 1 | Source: docA.pdf | Type: pdf]" in formatted
    assert "First piece of evidence." in formatted
    assert "[Chunk 2 | Source: docB.docx | Type: docx]" in formatted
    assert "Second piece of evidence." in formatted


def test_context_formatting_empty():
    assert _format_context([]) == "(no relevant context found)"


def test_graph_routing_stopping_conditions():
    # 1. Chunks are relevant -> go directly to generate
    state_relevant: GraphState = {
        "question": "q",
        "original_question": "q",
        "documents": [],
        "is_relevant": True,
        "retry_count": 0,
        "answer": "",
    }
    assert route_after_grading(state_relevant) == "generate"

    # 2. Not relevant but retries remain -> rewrite_query
    state_retry: GraphState = {
        "question": "q",
        "original_question": "q",
        "documents": [],
        "is_relevant": False,
        "retry_count": 0,
        "answer": "",
    }
    assert route_after_grading(state_retry) == "rewrite_query"

    # 3. Not relevant and max retries hit -> generate fallback
    state_exhausted: GraphState = {
        "question": "q",
        "original_question": "q",
        "documents": [],
        "is_relevant": False,
        "retry_count": MAX_RETRIES,
        "answer": "",
    }
    assert route_after_grading(state_exhausted) == "generate"
