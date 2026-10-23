"""
embedding.py

Provides the embedding model used to convert text chunks into vectors
for the semantic (FAISS) side of the Hybrid RAG pipeline.

Model: sentence-transformers/all-MiniLM-L6-v2
  - Fast, lightweight, 384-dim embeddings, strong general-purpose default.

Device: auto-detects CUDA if available, falls back to CPU.
You can override this with the EMBEDDING_DEVICE env var ("cpu" or "cuda").

Usage:
    from embedding import get_embedding_model

    embeddings = get_embedding_model()
    vector = embeddings.embed_query("some text")
    vectors = embeddings.embed_documents(["text 1", "text 2"])
"""

import os
import logging

os.environ["TQDM_DISABLE"] = "1"
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    import huggingface_hub.utils
    huggingface_hub.utils.disable_progress_bars()
except Exception:
    pass

try:
    import transformers.utils.logging as tf_logging
    tf_logging.disable_progress_bar()
    tf_logging.set_verbosity_error()
except Exception:
    pass

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


_model_cache = {}


def _detect_device() -> str:
    """
    Decide whether to run embeddings on GPU (CUDA) or CPU.

    Priority:
      1. EMBEDDING_DEVICE env var, if set ("cpu" or "cuda")
      2. Auto-detect via torch.cuda.is_available() and test allocation
      3. Fallback to "cpu"
    """
    override = os.getenv("EMBEDDING_DEVICE")
    if override in ("cpu", "cuda"):
        if override == "cuda":
            try:
                import torch
                if torch.cuda.is_available():
                    torch.zeros(1, device="cuda")
                    return "cuda"
                else:
                    logger.debug("CUDA requested via .env but not available; falling back to CPU.")
                    return "cpu"
            except Exception as e:
                logger.debug(f"CUDA check failed ({e}); falling back to CPU.")
                return "cpu"
        return "cpu"

    try:
        import torch

        if torch.cuda.is_available():
            torch.zeros(1, device="cuda")
            logger.info("CUDA GPU detected and verified — using GPU for embeddings.")
            return "cuda"
    except Exception as e:
        logger.debug(f"CUDA initialization skipped ({e}) — using CPU.")

    logger.debug("Using CPU for embeddings.")
    return "cpu"


def get_embedding_model(
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = None,
    normalize_embeddings: bool = True,
) -> HuggingFaceEmbeddings:
    """
    Load and return a HuggingFace embedding model wrapped for LangChain.
    Reuses cached instance if already loaded.

    Args:
        model_name: HuggingFace model repo name.
        device: "cpu" or "cuda". If None, auto-detects.
        normalize_embeddings: L2-normalize vectors — recommended for
            cosine-similarity search in FAISS.

    Returns:
        A LangChain-compatible HuggingFaceEmbeddings instance.
    """
    resolved_device = device or _detect_device()
    cache_key = (model_name, resolved_device, normalize_embeddings)

    if cache_key in _model_cache:
        return _model_cache[cache_key]

    logger.debug(f"Loading embedding model '{model_name}' on device='{resolved_device}'...")

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": resolved_device},
            encode_kwargs={"normalize_embeddings": normalize_embeddings},
        )
    except OSError as e:
        logger.debug(f"Memory/paging allocation warning ({e}). Limiting PyTorch threads and falling back to CPU...")
        try:
            import torch
            torch.set_num_threads(2)
        except Exception:
            pass
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": normalize_embeddings},
        )
    except Exception as e:
        if resolved_device != "cpu":
            logger.warning(f"Failed to load on {resolved_device} ({e}). Retrying on CPU...")
            embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": normalize_embeddings},
            )
        else:
            raise

    logger.info("Embedding model loaded successfully.")
    _model_cache[cache_key] = embeddings
    return embeddings


# ---------------------------------------------------------------------------
# Manual test run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    embedding_model = get_embedding_model()

    sample_texts = [
        "LangChain helps build LLM-powered applications.",
        "FAISS is a library for efficient vector similarity search.",
    ]

    vectors = embedding_model.embed_documents(sample_texts)

    print(f"\nGenerated {len(vectors)} embedding(s).")
    print(f"Embedding dimension: {len(vectors[0])}")
    print(f"First 5 values of vector 1: {vectors[0][:5]}")