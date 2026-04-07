"""
Provider factory — instantiates the correct provider based on secrets.toml config.

Config lives in .streamlit/secrets.toml:

  [providers]
  stt        = "assemblyai"    # or "deepgram" to roll back
  llm        = "cerebras"      # or "groq" or "openai" to roll back
  embeddings = "voyage"        # or "openai" to roll back
  ocr        = "gemini"        # or "none" to disable
  generation = "groq"          # for Content Studio

Module-level singletons: providers are instantiated once and reused.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Module-level singletons (lazy-init)
_stt  = None
_llm  = None
_emb  = None
_ocr  = None
_gen  = None


def _secret(key: str, default: str = "") -> str:
    """Read from st.secrets if available, fall back to env var."""
    try:
        import streamlit as st
        return st.secrets.get(key, os.environ.get(key, default))
    except Exception:
        return os.environ.get(key, default)


def _provider_cfg(key: str, default: str) -> str:
    """Read from [providers] section of secrets.toml."""
    try:
        import streamlit as st
        return st.secrets.get("providers", {}).get(key, default)
    except Exception:
        return default


def get_stt():
    """Return the configured STT provider (singleton)."""
    global _stt
    if _stt is not None:
        return _stt

    provider = _provider_cfg("stt", "assemblyai")
    if provider == "assemblyai":
        from providers.stt import AssemblyAIProvider
        _stt = AssemblyAIProvider(api_key=_secret("ASSEMBLYAI_API_KEY"))
    elif provider == "deepgram":
        # Rollback: re-wrap DeepgramTranscriber to match new interface
        from providers._deepgram_adapter import DeepgramAdapter
        _stt = DeepgramAdapter(api_key=_secret("DEEPGRAM_API_KEY"))
    else:
        raise ValueError(f"Unknown STT provider: {provider}")
    return _stt


def get_llm():
    """Return the configured extraction LLM provider (singleton)."""
    global _llm
    if _llm is not None:
        return _llm

    provider = _provider_cfg("llm", "cerebras")
    if provider == "cerebras":
        from providers.llm import CerebrasProvider
        _llm = CerebrasProvider(api_key=_secret("CEREBRAS_API_KEY"))
    elif provider == "groq":
        from providers.llm import GroqLLMProvider
        _llm = GroqLLMProvider(api_key=_secret("GROQ_API_KEY"))
    elif provider == "openai":
        # Rollback: use existing extraction.py function directly
        _llm = None   # caller should fall back to analysis.extraction.extract_message_units()
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
    return _llm


def get_embeddings():
    """Return the configured embedding provider (singleton)."""
    global _emb
    if _emb is not None:
        return _emb

    provider = _provider_cfg("embeddings", "voyage")
    if provider == "voyage":
        from providers.embeddings import VoyageProvider
        _emb = VoyageProvider(api_key=_secret("VOYAGE_API_KEY"))
    elif provider == "openai":
        # Rollback: return None — caller uses analysis.embeddings.embed_batch() directly
        _emb = None
    else:
        raise ValueError(f"Unknown embeddings provider: {provider}")
    return _emb


def get_ocr() -> Optional[object]:
    """Return the configured OCR provider, or None if disabled."""
    global _ocr
    if _ocr is not None:
        return _ocr

    provider = _provider_cfg("ocr", "gemini")
    if provider == "none":
        return None
    if provider == "gemini":
        from providers.ocr import GeminiOCRProvider
        # secrets.toml uses GOOGLE_API_KEY (Google AI Studio key)
        _ocr = GeminiOCRProvider(api_key=_secret("GOOGLE_API_KEY") or _secret("GEMINI_API_KEY"))
    else:
        return None
    return _ocr


def get_generation_llm():
    """Return the LLM provider used for Content Studio generation (singleton)."""
    global _gen
    if _gen is not None:
        return _gen

    provider = _provider_cfg("generation", "groq")
    if provider == "groq":
        from providers.llm import GroqLLMProvider
        _gen = GroqLLMProvider(api_key=_secret("GROQ_API_KEY"))
    elif provider == "cerebras":
        from providers.llm import CerebrasProvider
        _gen = CerebrasProvider(api_key=_secret("CEREBRAS_API_KEY"))
    else:
        raise ValueError(f"Unknown generation provider: {provider}")
    return _gen


def reset_singletons():
    """Reset all singletons — useful in tests or after secrets change."""
    global _stt, _llm, _emb, _ocr, _gen
    _stt = _llm = _emb = _ocr = _gen = None
