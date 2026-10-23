"""
storage/history.py

Lightweight persistent conversation history for the Hybrid RAG application.
Stores conversations as individual JSON files in a dedicated storage directory
for atomic, human-readable, and corruption-free persistence.
"""

import json
import logging
import re
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

STORAGE_DIR = Path("./storage/conversations")


def _get_storage_dir() -> Path:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    return STORAGE_DIR


def generate_title_from_query(query: str, max_chars: int = 42) -> str:
    """
    Generate a clean, professional, readable title from the first user query.
    Strips leading filler phrases ("can you explain", "what is", etc.),
    capitalizes neatly, and truncates on word boundaries.
    """
    cleaned = query.strip()
    # Normalize internal whitespace
    cleaned = re.sub(r"\s+", " ", cleaned)

    # Remove common conversational query prefixes
    filler_patterns = [
        r"^(can you (please )?(tell me|explain|describe|show me))\s+(about\s+)?(the\s+|a\s+|an\s+)?",
        r"^(what is|what are|what do you mean by)\s+(the\s+|a\s+|an\s+)?",
        r"^(how does|how do|how can I|how to)\s+(the\s+|a\s+|an\s+)?",
        r"^(tell me about|explain to me|give me details on)\s+(the\s+|a\s+|an\s+)?",
        r"^(please )?(find|summarize|list|compare)\s+(the\s+|a\s+|an\s+)?",
    ]
    for pattern in filler_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()

    if not cleaned:
        cleaned = query.strip()

    # Capitalize the first letter
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]

    # Remove trailing punctuation (question marks, periods, commas)
    cleaned = re.sub(r"[?!.,;:]+$", "", cleaned).strip()

    # Truncate at word boundary if too long
    if len(cleaned) > max_chars:
        truncated = cleaned[:max_chars].rsplit(" ", 1)[0]
        return (truncated or cleaned[:max_chars]) + "…"

    return cleaned or "New Conversation"


def create_conversation(title: str = "New Chat") -> dict:
    """Create a new conversation, save to disk, and return its dict."""
    storage_dir = _get_storage_dir()
    cid = str(uuid.uuid4())
    now_iso = datetime.now(timezone.utc).isoformat()

    conv = {
        "id": cid,
        "title": title,
        "created_at": now_iso,
        "updated_at": now_iso,
        "messages": [],
    }

    file_path = storage_dir / f"{cid}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(conv, f, indent=2, ensure_ascii=False)

    logger.info(f"Created conversation {cid}: '{title}'")
    return conv


def get_conversation(cid: str) -> Optional[dict]:
    """Load a conversation by id from disk."""
    if not cid:
        return None
    file_path = _get_storage_dir() / f"{cid}.json"
    if not file_path.exists():
        return None

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error reading conversation {cid}: {e}")
        return None


def save_conversation(conv: dict) -> None:
    """Save an existing conversation dict to disk."""
    cid = conv.get("id")
    if not cid:
        raise ValueError("Conversation must have an 'id'")

    conv["updated_at"] = datetime.now(timezone.utc).isoformat()
    file_path = _get_storage_dir() / f"{cid}.json"

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(conv, f, indent=2, ensure_ascii=False)


def list_conversations() -> list[dict]:
    """
    Return all saved conversations, sorted by updated_at descending.
    Lightweight: loads summary info.
    """
    storage_dir = _get_storage_dir()
    conversations = []

    for file_path in storage_dir.glob("*.json"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                conversations.append(data)
        except Exception as e:
            logger.warning(f"Skipping corrupted conversation file {file_path.name}: {e}")

    # Sort newest first
    conversations.sort(
        key=lambda c: c.get("updated_at") or c.get("created_at") or "",
        reverse=True,
    )
    return conversations


def add_message(
    cid: str,
    role: str,
    content: str,
    sources: Optional[list] = None,
    retrieval_details: Optional[dict] = None,
    timing_s: Optional[float] = None,
) -> dict:
    """
    Append a message to a conversation. If it's the first user message,
    automatically update the conversation title.
    """
    conv = get_conversation(cid)
    if not conv:
        conv = create_conversation()
        cid = conv["id"]

    msg = {
        "id": str(uuid.uuid4()),
        "role": role,
        "content": content,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sources": sources or [],
        "retrieval_details": retrieval_details or {},
        "timing_s": timing_s,
    }

    # Auto-title on first user message if current title is default
    if role == "user" and (conv.get("title") in ("New Chat", "New Conversation") or not conv.get("messages")):
        conv["title"] = generate_title_from_query(content)

    conv["messages"].append(msg)
    save_conversation(conv)
    return conv


def rename_conversation(cid: str, new_title: str) -> bool:
    """Rename a conversation."""
    conv = get_conversation(cid)
    if not conv:
        return False
    conv["title"] = new_title.strip() or "Untitled Chat"
    save_conversation(conv)
    return True


def delete_conversation(cid: str) -> bool:
    """Delete a conversation file from disk."""
    file_path = _get_storage_dir() / f"{cid}.json"
    if file_path.exists():
        try:
            file_path.unlink()
            logger.info(f"Deleted conversation {cid}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete conversation {cid}: {e}")
            return False
    return False


def clear_all_conversations() -> int:
    """Delete all conversations. Returns count of deleted conversations."""
    storage_dir = _get_storage_dir()
    count = 0
    for file_path in storage_dir.glob("*.json"):
        try:
            file_path.unlink()
            count += 1
        except Exception as e:
            logger.warning(f"Error removing {file_path.name}: {e}")
    logger.info(f"Cleared {count} conversations.")
    return count


def group_conversations_by_date(conversations: list[dict]) -> dict[str, list[dict]]:
    """
    Group conversations into human-friendly sections:
      - Today
      - Yesterday
      - Previous 7 Days
      - Older
    """
    grouped = {
        "Today": [],
        "Yesterday": [],
        "Previous 7 Days": [],
        "Older": [],
    }

    now = datetime.now(timezone.utc)
    today_date = now.date()
    yesterday_date = today_date - timedelta(days=1)
    seven_days_ago = today_date - timedelta(days=7)

    for conv in conversations:
        ts_str = conv.get("updated_at") or conv.get("created_at")
        if not ts_str:
            grouped["Older"].append(conv)
            continue

        try:
            # Handle ISO format with or without Z/offset
            if ts_str.endswith("Z"):
                ts_str = ts_str[:-1] + "+00:00"
            conv_date = datetime.fromisoformat(ts_str).date()
        except Exception:
            grouped["Older"].append(conv)
            continue

        if conv_date == today_date:
            grouped["Today"].append(conv)
        elif conv_date == yesterday_date:
            grouped["Yesterday"].append(conv)
        elif conv_date > seven_days_ago:
            grouped["Previous 7 Days"].append(conv)
        else:
            grouped["Older"].append(conv)

    # Return only non-empty groups
    return {k: v for k, v in grouped.items() if v}
