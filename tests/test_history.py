"""
tests/test_history.py

Unit tests for persistent conversation storage:
- CRUD operations
- Auto-titling logic
- Date grouping
- State isolation & persistence
"""

import shutil
import tempfile
from pathlib import Path
import pytest

import storage.history as hist


@pytest.fixture(autouse=True)
def temp_storage(monkeypatch):
    """Isolate storage directory for each test."""
    temp_dir = Path(tempfile.mkdtemp())
    monkeypatch.setattr(hist, "STORAGE_DIR", temp_dir)
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_create_and_get_conversation():
    conv = hist.create_conversation("My First Chat")
    assert conv["title"] == "My First Chat"
    assert "id" in conv
    assert len(conv["messages"]) == 0

    fetched = hist.get_conversation(conv["id"])
    assert fetched is not None
    assert fetched["id"] == conv["id"]
    assert fetched["title"] == "My First Chat"


def test_auto_generate_title():
    t1 = hist.generate_title_from_query("What is the PSK modulation technique?")
    assert t1 == "PSK modulation technique"

    t2 = hist.generate_title_from_query("Can you please tell me about Kubernetes cluster architecture and deployment strategies?")
    assert len(t2) <= 45
    assert not t2.startswith("Can you please")

    t3 = hist.generate_title_from_query("simple query")
    assert t3 == "Simple query"


def test_add_message_and_auto_titling():
    conv = hist.create_conversation("New Chat")
    cid = conv["id"]

    # First user message should automatically set a meaningful title
    hist.add_message(cid, role="user", content="How does OFDM work in wireless communication?")
    updated = hist.get_conversation(cid)
    assert len(updated["messages"]) == 1
    assert updated["messages"][0]["role"] == "user"
    assert "OFDM" in updated["title"]

    # Assistant message should append cleanly
    hist.add_message(
        cid,
        role="assistant",
        content="OFDM divides spectrum into orthogonal subcarriers.",
        sources=[{"filename": "notes.pdf", "page": 1}],
    )
    updated2 = hist.get_conversation(cid)
    assert len(updated2["messages"]) == 2
    assert updated2["messages"][1]["role"] == "assistant"
    assert len(updated2["messages"][1]["sources"]) == 1


def test_rename_and_delete_conversation():
    conv = hist.create_conversation("Original Title")
    cid = conv["id"]

    assert hist.rename_conversation(cid, "Updated Title") is True
    assert hist.get_conversation(cid)["title"] == "Updated Title"

    assert hist.delete_conversation(cid) is True
    assert hist.get_conversation(cid) is None


def test_clear_all_conversations():
    hist.create_conversation("Chat 1")
    hist.create_conversation("Chat 2")
    hist.create_conversation("Chat 3")

    assert len(hist.list_conversations()) == 3
    deleted_count = hist.clear_all_conversations()
    assert deleted_count == 3
    assert len(hist.list_conversations()) == 0


def test_date_grouping():
    conv1 = hist.create_conversation("Chat Today")
    groups = hist.group_conversations_by_date([conv1])
    assert "Today" in groups
    assert len(groups["Today"]) == 1
