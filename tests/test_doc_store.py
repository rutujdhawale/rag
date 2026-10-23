"""
tests/test_doc_store.py

Unit tests for document store and catalog tracking.
"""

import shutil
import tempfile
from pathlib import Path
import pytest

import storage.doc_store as ds


@pytest.fixture(autouse=True)
def temp_catalog(monkeypatch):
    temp_dir = Path(tempfile.mkdtemp())
    catalog_file = temp_dir / "catalog.json"
    monkeypatch.setattr(ds, "CATALOG_FILE", catalog_file)
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_format_file_size():
    assert ds.format_file_size(500) == "500 B"
    assert ds.format_file_size(2048) == "2.0 KB"
    assert ds.format_file_size(5 * 1024 * 1024) == "5.0 MB"


def test_record_and_get_document():
    doc = ds.record_document_indexed(
        filename="test_manual.pdf",
        file_type="pdf",
        size_bytes=1048576,
        chunk_count=24,
    )
    assert doc["filename"] == "test_manual.pdf"
    assert doc["chunk_count"] == 24
    assert doc["size_formatted"] == "1.0 MB"
    assert doc["status"] == "Indexed"
