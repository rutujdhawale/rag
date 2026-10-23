"""
core/config.py

Centralized configuration and environment loading for Hybrid RAG.
Provides defaults and validation for settings across UI and Pipeline.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


@dataclass
class AppConfig:
    data_dir: str = os.getenv("DATA_DIR", "./data")
    faiss_index_dir: str = os.getenv("FAISS_INDEX_DIR", "./faiss_index")

    # Retrieval defaults
    semantic_weight: float = float(os.getenv("SEMANTIC_WEIGHT", "0.6"))
    keyword_weight: float = float(os.getenv("KEYWORD_WEIGHT", "0.4"))
    top_k: int = int(os.getenv("RETRIEVER_TOP_K", "4"))
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # Model defaults
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_provider: str = os.getenv("LLM_PROVIDER", "groq" if os.getenv("GROQ_API_KEY") else "openrouter")
    openrouter_model: str = os.getenv("OPENROUTER_MODEL", "google/gemma-3-27b-it:free")
    groq_model: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))

    # LangGraph Self-Correction
    enable_self_correction: bool = True
    max_retries: int = int(os.getenv("MAX_RETRIEVAL_RETRIES", "2"))


def get_default_config() -> AppConfig:
    return AppConfig()
