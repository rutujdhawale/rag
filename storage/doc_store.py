"""
storage/doc_store.py

Document catalog tracking for files indexed in the Hybrid RAG pipeline.
Maintains persistent document metadata (size, chunks, indexing date, status)
and synchronizes with disk files in the DATA_DIR.
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

CATALOG_FILE = Path("./storage/documents_catalog.json")


def format_file_size(size_bytes: int) -> str:
    """Format bytes into a human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


def _load_raw_catalog() -> dict[str, dict]:
    CATALOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not CATALOG_FILE.exists():
        return {}
    try:
        with open(CATALOG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Error loading document catalog: {e}")
        return {}


def _save_raw_catalog(catalog: dict[str, dict]) -> None:
    CATALOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CATALOG_FILE, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)


def record_document_indexed(
    filename: str,
    file_type: str,
    size_bytes: int,
    chunk_count: int,
    status: str = "Indexed",
) -> dict:
    """Record or update metadata for an indexed document."""
    catalog = _load_raw_catalog()
    doc_info = {
        "filename": filename,
        "file_type": file_type.lower().lstrip("."),
        "size_bytes": size_bytes,
        "size_formatted": format_file_size(size_bytes),
        "chunk_count": chunk_count,
        "status": status,
        "indexed_at": datetime.now(timezone.utc).isoformat(),
    }
    catalog[filename] = doc_info
    _save_raw_catalog(catalog)
    return doc_info


def get_document_catalog(data_dir: str = "./data") -> list[dict]:
    """
    Get the list of all documents present in data_dir, enriched with
    catalog metadata (chunks, indexed date) if available.
    """
    catalog = _load_raw_catalog()
    dir_path = Path(data_dir)
    results = []

    if dir_path.exists() and dir_path.is_dir():
        for file_path in dir_path.iterdir():
            if not file_path.is_file():
                continue

            name = file_path.name
            size = file_path.stat().st_size
            suffix = file_path.suffix.lstrip(".").lower()

            if name in catalog:
                info = catalog[name].copy()
                info["size_bytes"] = size
                info["size_formatted"] = format_file_size(size)
                info["exists_on_disk"] = True
            else:
                info = {
                    "filename": name,
                    "file_type": suffix,
                    "size_bytes": size,
                    "size_formatted": format_file_size(size),
                    "chunk_count": 0,
                    "status": "Detected on disk",
                    "indexed_at": None,
                    "exists_on_disk": True,
                }
            results.append(info)

    # Sort alphabetically by filename
    results.sort(key=lambda d: d["filename"].lower())
    return results


def remove_document(filename: str, data_dir: str = "./data") -> bool:
    """Remove a document from disk and catalog."""
    catalog = _load_raw_catalog()
    if filename in catalog:
        del catalog[filename]
        _save_raw_catalog(catalog)

    disk_path = Path(data_dir) / filename
    if disk_path.exists() and disk_path.is_file():
        try:
            disk_path.unlink()
            logger.info(f"Deleted {filename} from disk.")
            return True
        except Exception as e:
            logger.error(f"Failed to remove file {filename}: {e}")
            return False
    return True
