"""
Embedding providers.

VoyageProvider — primary (voyage-3-lite, 512-dim, 200M tokens free at signup)

⚠️  Changing embedding provider changes vector dimensions.
    Existing embeddings must be re-computed after switching.
    See core/db.py migration for FLOAT[512] column resize.
"""

from __future__ import annotations

import logging
from typing import List

logger = logging.getLogger(__name__)

VOYAGE_MODEL   = "voyage-3-lite"
VOYAGE_DIMS    = 512
VOYAGE_COST_PER_TOKEN = 0.000018 / 1_000  # $0.000018/1M tokens


class VoyageProvider:
    """
    Local-quality embeddings via Voyage AI.

    Model:     voyage-3-lite (512-dim)
    Quality:   outperforms text-embedding-3-small on retrieval benchmarks (MTEB)
    Free tier: 200M tokens at signup (~2,000 full pipeline runs)
    Cost:      $0.000018/1M tokens after free tier

    Install: pip install voyageai
    Signup:  voyageai.com
    """

    def __init__(self, api_key: str, model: str = VOYAGE_MODEL):
        if not api_key:
            raise ValueError("VOYAGE_API_KEY is required")
        import voyageai
        self._client = voyageai.Client(api_key=api_key)
        self._model = model

    @property
    def dimensions(self) -> int:
        return VOYAGE_DIMS

    def embed(self, text: str) -> List[float]:
        """Embed a single string. Returns 512-dim float list."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str], rate_per_min: int = 3) -> List[List[float]]:
        """
        Batch-embed texts.  Voyage supports batches up to 128 items.
        Automatically splits larger lists into sub-batches.

        rate_per_min: API calls per minute.  Default 3 = Voyage free-tier limit.
        After adding a payment method the free tier becomes 300+ RPM; pass
        rate_per_min=300 (or higher) once your billing is set up to skip the delay.
        """
        import time
        if not texts:
            return []

        all_embeddings: List[List[float]] = []
        batch_size = 128
        delay = 60.0 / max(rate_per_min, 1)  # seconds between API calls

        for idx, i in enumerate(range(0, len(texts), batch_size)):
            if idx > 0:
                time.sleep(delay)
            batch = texts[i : i + batch_size]
            result = self._client.embed(
                batch,
                model=self._model,
                input_type="document",
            )
            all_embeddings.extend(result.embeddings)

        return all_embeddings
