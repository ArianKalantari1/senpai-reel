"""
OCR provider — frame text extraction via Gemini Flash 2.0.

Replaces PaddleOCR (local) with a cloud call that:
  - Extracts all visible text in reading order
  - Understands visual context (what the creator is demonstrating)
  - Handles rotated text, stylised fonts, and split-screen layouts

Free tier: 1,500 req/day, 15 req/min (Google AI Studio)
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_OCR_PROMPT = (
    "You are analysing a frame from an Instagram reel. "
    "1. List ALL visible text exactly as it appears, in reading order (top to bottom, left to right). "
    "2. If the creator is demonstrating a visual concept (chart, diagram, list, step-by-step), "
    "briefly describe what is shown. "
    "Return only the extracted content — no preamble, no commentary."
)


class GeminiOCRProvider:
    """
    Frame text extraction via Google Gemini Flash 2.0.

    Free tier: 1,500 req/day  (enough for 300 reels at 5 frames each)
    Signup:    aistudio.google.com (Google account)

    Install: pip install google-generativeai
    """

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required")
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model)

    def extract(self, image_path: str, post_id: str = "") -> str:
        """
        Extract text from a single frame image.

        Args:
            image_path: absolute path to a JPEG or PNG frame
            post_id:    for logging only

        Returns:
            Extracted text as a plain string. Empty string if nothing found.
        """
        path = Path(image_path)
        if not path.exists():
            logger.warning("Frame not found for OCR: %s (post %s)", image_path, post_id)
            return ""

        with open(path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        suffix = path.suffix.lower()
        mime_type = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png"

        response = self._model.generate_content([
            _OCR_PROMPT,
            {"mime_type": mime_type, "data": image_data},
        ])

        text = response.text.strip() if response.text else ""
        return text
