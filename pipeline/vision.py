"""
vision.py

Vision / multimodal module for the Hybrid RAG pipeline.

Uses OpenRouter's vision-capable model (Gemma 3 27B supports image input)
to describe images, charts, graphs, and diagrams extracted from PDFs.

The descriptions are stored as text Document chunks, making visual content
searchable and retrievable just like regular text.

Usage:
    from pipeline.vision import describe_image

    description = describe_image(image_bytes, context="Page 3 of report.pdf")
"""

import os
import io
import base64
import logging
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
VISION_MODEL = os.getenv("VISION_MODEL", "google/gemma-3-27b-it:free")


def describe_image(
    image_bytes: bytes,
    context: str = "",
    model: str = VISION_MODEL,
) -> Optional[str]:
    """
    Send an image to a vision-capable LLM and get a detailed text description.

    This handles charts, graphs, diagrams, tables, and any visual content.

    Args:
        image_bytes: Raw image bytes (PNG, JPEG, etc.)
        context: Optional context string (e.g. "Page 3 of financial_report.pdf")
        model: Vision model to use via OpenRouter.

    Returns:
        Text description of the image content, or None if it fails.
    """
    if not OPENROUTER_API_KEY:
        logger.warning("OPENROUTER_API_KEY not set — cannot describe images via vision model.")
        return _fallback_ocr(image_bytes)

    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage

        llm = ChatOpenAI(
            model=model,
            max_tokens=1024,
            temperature=0.1,
            request_timeout=25,
            openai_api_key=OPENROUTER_API_KEY,
            openai_api_base="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "http://localhost",
                "X-Title": "Hybrid RAG Pipeline",
            },
        )

        b64_image = base64.b64encode(image_bytes).decode("utf-8")
        mime_type = _detect_mime_type(image_bytes)
        context_hint = f"\n\nContext: This image was extracted from {context}." if context else ""

        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": (
                        "Describe this image clearly and thoroughly. If it contains:\n"
                        "- A chart or graph: state the type, axes, trends, and key values\n"
                        "- A diagram or flowchart: list components and dataflow\n"
                        "- A table: extract the key rows and columns\n"
                        "- Text: transcribe it accurately\n"
                        f"{context_hint}"
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{b64_image}",
                    },
                },
            ]
        )

        logger.info(f"Sending image to OpenRouter vision model ({model})...")
        response = llm.invoke([message])
        description = response.content.strip() if response and response.content else ""

        if description:
            logger.info(f"Vision model returned description ({len(description)} chars).")
            return description
        else:
            return _try_groq_vision(image_bytes, context) or _fallback_ocr(image_bytes)

    except Exception as e:
        logger.warning(f"OpenRouter vision failed: {e}. Trying Groq vision / OCR fallback...")
        return _try_groq_vision(image_bytes, context) or _fallback_ocr(image_bytes)


def _try_groq_vision(image_bytes: bytes, context: str = "") -> Optional[str]:
    """Fallback to Groq vision model if available."""
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        return None
    try:
        from langchain_groq import ChatGroq
        from langchain_core.messages import HumanMessage

        groq_llm = ChatGroq(
            model="llama-3.2-11b-vision-preview",
            temperature=0.1,
            api_key=groq_key,
            request_timeout=25,
        )

        b64_image = base64.b64encode(image_bytes).decode("utf-8")
        mime_type = _detect_mime_type(image_bytes)
        context_hint = f"\n\nContext: This image was extracted from {context}." if context else ""

        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": f"Describe this image concisely and extract all visible text and charts.{context_hint}",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{b64_image}",
                    },
                },
            ]
        )
        logger.info("Calling Groq vision fallback (llama-3.2-11b-vision-preview)...")
        res = groq_llm.invoke([message])
        if res and res.content:
            return res.content.strip()
    except Exception as e:
        logger.warning(f"Groq vision fallback skipped/failed: {e}")
    return None


def _detect_mime_type(image_bytes: bytes) -> str:
    """Detect MIME type from image magic bytes."""
    if image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
        return "image/png"
    elif image_bytes[:2] == b'\xff\xd8':
        return "image/jpeg"
    elif image_bytes[:4] == b'GIF8':
        return "image/gif"
    elif image_bytes[:4] == b'RIFF' and image_bytes[8:12] == b'WEBP':
        return "image/webp"
    elif image_bytes[:2] in (b'BM',):
        return "image/bmp"
    else:
        return "image/png"  # default assumption


def _fallback_ocr(image_bytes: bytes) -> Optional[str]:
    """
    Fallback: use pytesseract OCR if the vision model is unavailable.
    Only extracts text — can't understand charts/graphs semantically.
    """
    try:
        from PIL import Image
        import pytesseract

        image = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(image).strip()

        if text:
            logger.info(f"OCR fallback extracted {len(text)} chars of text.")
            return f"[OCR extracted text from image]\n{text}"
        else:
            logger.warning("OCR fallback: no text found in image.")
            return None

    except ImportError:
        logger.warning("pytesseract not available for OCR fallback.")
        return None
    except Exception as e:
        logger.warning(f"OCR fallback failed: {e}")
        return None


def describe_image_batch(
    images: list,
    context: str = "",
) -> list:
    """
    Describe multiple images. Returns a list of (description, index) tuples.

    Args:
        images: List of (image_bytes, page_number) tuples.
        context: Source file context string.

    Returns:
        List of (description_text, page_number) tuples for successfully described images.
    """
    results = []
    for image_bytes, page_num in images:
        desc = describe_image(
            image_bytes,
            context=f"{context}, page {page_num}",
        )
        if desc:
            results.append((desc, page_num))

    logger.info(f"Described {len(results)}/{len(images)} images from {context}.")
    return results
