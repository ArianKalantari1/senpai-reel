# Local Provider Stack — Technical Specification
*Machine: Apple M1, 16 GB unified memory | Date: 7 April 2026*

---

## 1. Design principle

Replace every paid API call with a specialized local model.  
Do **not** use one model for everything — use the right tool for each layer.

```
Current (paid)                     Target (local)
──────────────────────────────────────────────────────────────
Audio → Deepgram Nova-2        →   faster-whisper (large-v3 INT8)
Frames → (nothing)             →   PaddleOCR (new per-frame layer)
Text → GPT-4o-mini (extract)   →   Qwen2.5-VL 7B via Ollama
Text → text-embedding-3-small  →   qwen3-embedding or all-minilm via Ollama
Text → GPT-4o-mini (generate)  →   Gemma3:12b via Ollama  (secondary)
```

All local models are served through **Ollama** as a unified local API.  
`faster-whisper` runs in-process (Python library, not via Ollama — it is faster that way).  
`PaddleOCR` runs in-process (Python library).

---

## 2. Hardware constraints (M1, 16 GB)

Apple M1 uses unified memory — GPU and CPU share the same 16 GB pool.  
The OS + Streamlit typically use ~4–5 GB at rest, leaving ~11 GB for models.

Model size budget at Q4 quantization (approx):

| Model | Quantized size | Status on M1 16 GB |
|-------|---------------|-------------------|
| faster-whisper large-v3 INT8 | ~1.5 GB | ✅ comfortable |
| qwen2.5vl:7b (Q4) | ~4.5 GB | ✅ comfortable |
| qwen2.5vl:3b (Q4) | ~2.5 GB | ✅ very comfortable |
| gemma3:12b (Q4) | ~7.5 GB | ⚠️ tight — don't load with Qwen simultaneously |
| gemma3:4b (Q4) | ~3.0 GB | ✅ load alongside Qwen if needed |
| qwen3-embedding (0.6b) | ~0.5 GB | ✅ trivial |

**Rule of thumb:** never load Qwen VL and Gemma 12B at the same time.  
Use them in sequence — Ollama unloads models when not in use (default: after 5 minutes idle).

---

## 3. Exact package versions

Add these to `requirements.txt`. Do not install all at once — follow the migration order in Section 6.

```
# -- Local STT
faster-whisper==1.1.1

# -- Local OCR
paddleocr==2.9.1
paddlepaddle==2.6.2

# -- Ollama client (for VLM + embeddings)
ollama==0.4.7

# -- Sentence Transformers (fallback / testing embeddings without Ollama)
sentence-transformers==3.4.1
```

**Ollama itself** (the server) is installed separately — not via pip:
```bash
brew install ollama
```

**Models to pull after Ollama is installed:**
```bash
ollama pull qwen2.5vl:7b          # primary VLM for extraction (4.5 GB)
ollama pull qwen3-embedding        # local embeddings (500 MB)
ollama pull gemma3:4b              # secondary generator (3 GB)
# Optional: gemma3:12b if you want higher quality generation and run it alone
# ollama pull gemma3:12b
```

---

## 4. Provider interfaces

Create `providers/` as a new top-level package.  
Each interface is a simple Python class with one mandatory method.  
Concrete implementations live in the same file as their interface.

### 4.1 File layout

```
providers/
    __init__.py
    stt.py           # SpeechToTextProvider + FasterWhisperProvider
    ocr.py           # OCRProvider + PaddleOCRProvider
    llm.py           # LLMExtractionProvider + OllamaQwenVLProvider + OllamaGemmaProvider
    embeddings.py    # EmbeddingProvider + OllamaEmbeddingProvider + SentenceTransformerProvider
    factory.py       # get_stt(), get_ocr(), get_llm(), get_embeddings() — reads config
```

---

### 4.2 `providers/stt.py`

```python
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List

@dataclass
class WordTimestamp:
    word: str
    start_sec: float
    end_sec: float
    confidence: float

@dataclass
class STTResult:
    transcript: str
    language: str
    duration_sec: float
    words: List[WordTimestamp] = field(default_factory=list)
    provider: str = ""
    model: str = ""
    cost_usd: float = 0.0          # always 0.0 for local providers


class SpeechToTextProvider(ABC):
    @abstractmethod
    def transcribe(self, audio_path: str, post_id: str) -> STTResult:
        """Transcribe a WAV file. Returns STTResult."""


class FasterWhisperProvider(SpeechToTextProvider):
    """
    Local in-process transcription using faster-whisper + CTranslate2.

    Args:
        model_size: "large-v3" recommended; "medium" for speed/memory tradeoff
        device:     "auto" lets CTranslate2 pick (CPU on M1, MPS on M1 if supported)
        compute_type: "int8" for smallest memory footprint; "float16" for accuracy
    """
    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "auto",
        compute_type: str = "int8",
    ):
        from faster_whisper import WhisperModel
        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self._model_size = model_size

    def transcribe(self, audio_path: str, post_id: str) -> STTResult:
        segments, info = self._model.transcribe(
            audio_path,
            language="en",           # lock to English; auto-detect costs extra inference
            beam_size=5,
            word_timestamps=True,
        )
        words = []
        full_text_parts = []
        for segment in segments:
            full_text_parts.append(segment.text)
            if segment.words:
                for w in segment.words:
                    words.append(WordTimestamp(
                        word=w.word,
                        start_sec=w.start,
                        end_sec=w.end,
                        confidence=w.probability,
                    ))
        return STTResult(
            transcript=" ".join(full_text_parts).strip(),
            language=info.language,
            duration_sec=info.duration,
            words=words,
            provider="faster-whisper",
            model=self._model_size,
            cost_usd=0.0,
        )
```

**Integration note:** `STTResult` has the same fields as the existing `TranscriptResult` in `processing/transcribe.py`. The migration task is to make `transcribe_post()` accept a `SpeechToTextProvider` and call `provider.transcribe()` instead of calling Deepgram directly.

---

### 4.3 `providers/ocr.py`

```python
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List
import numpy as np

@dataclass
class OCRResult:
    lines: List[str]          # extracted text lines in reading order
    raw_boxes: list           # raw PaddleOCR box output (for debugging)
    provider: str = ""

    @property
    def full_text(self) -> str:
        return " ".join(self.lines)


class OCRProvider(ABC):
    @abstractmethod
    def extract_text(self, image_path: str) -> OCRResult:
        """Extract text from a single image file. Returns OCRResult."""


class PaddleOCRProvider(OCRProvider):
    """
    In-process OCR using PaddleOCR (English mode, angle correction on).

    Install: pip install paddleocr paddlepaddle

    On first run it downloads ~300 MB of model weights to ~/.paddleocr/.
    Subsequent runs load from cache — startup takes ~2 seconds.
    """
    def __init__(self, lang: str = "en"):
        from paddleocr import PaddleOCR
        # use_angle_cls=True corrects rotated text (common in reels)
        self._ocr = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)

    def extract_text(self, image_path: str) -> OCRResult:
        result = self._ocr.ocr(image_path, cls=True)
        lines = []
        raw_boxes = []
        if result and result[0]:
            for line in result[0]:
                box, (text, confidence) = line
                if confidence > 0.5:   # discard low-confidence detections
                    lines.append(text)
                raw_boxes.append(line)
        return OCRResult(lines=lines, raw_boxes=raw_boxes, provider="paddle-ocr")
```

**Why OCR matters for this app:**  
Instagram reels typically burn captions, title cards, bullet points, and CTAs directly into the video as text overlays. The audio transcript captures what the creator *says* but misses what they *show*. OCR fills that gap — the merger of transcript + OCR text gives the VLM much richer input for extraction.

---

### 4.4 `providers/llm.py`

```python
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional
import json

@dataclass
class ExtractionResult:
    units: list          # list of raw dicts matching the message_units schema
    model: str = ""
    cost_usd: float = 0.0   # 0.0 for local providers


@dataclass
class GenerationResult:
    text: str
    model: str = ""
    tokens_used: int = 0
    cost_usd: float = 0.0


class LLMExtractionProvider(ABC):
    @abstractmethod
    def extract(
        self,
        transcript: str,
        ocr_text: str,          # new: OCR-sourced text from video frames
        caption: str,           # post caption from DB
        system_prompt: str,
    ) -> ExtractionResult:
        """Extract message units from combined transcript + OCR + caption."""

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1000,
    ) -> GenerationResult:
        """Generate text (captions, hooks, scripts)."""


class OllamaQwenVLProvider(LLMExtractionProvider):
    """
    Multimodal extraction using Qwen2.5-VL via Ollama.

    Model: qwen2.5vl:7b (7B param, Q4, ~4.5 GB VRAM — fits on M1 16 GB)
    Fallback: qwen2.5vl:3b if memory is tight

    Pull model first: ollama pull qwen2.5vl:7b
    """
    def __init__(
        self,
        model: str = "qwen2.5vl:7b",
        base_url: str = "http://localhost:11434",
    ):
        import ollama
        self._client = ollama.Client(host=base_url)
        self._model = model

    def extract(
        self,
        transcript: str,
        ocr_text: str,
        caption: str,
        system_prompt: str,
        frame_paths: Optional[List[str]] = None,   # optional: sampled frames as images
    ) -> ExtractionResult:
        # Build user content — text-only if no frames provided, multimodal if frames exist
        text_input = f"""Transcript: {transcript[:3000]}

On-screen text (OCR from video frames): {ocr_text[:1000] or 'none detected'}

Caption: {caption[:500] or 'none'}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": text_input},
        ]

        # If frames are available, attach up to 3 sampled frames as images
        if frame_paths:
            import base64
            images = []
            for fp in frame_paths[:3]:
                with open(fp, "rb") as f:
                    images.append(base64.b64encode(f.read()).decode())
            messages[-1]["images"] = images

        response = self._client.chat(
            model=self._model,
            messages=messages,
            format="json",              # Ollama structured output
            options={"temperature": 0.2},
        )
        raw = response["message"]["content"]
        try:
            parsed = json.loads(raw)
            units = parsed.get("units", [])
        except json.JSONDecodeError:
            units = []

        return ExtractionResult(units=units, model=self._model, cost_usd=0.0)

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 1000) -> GenerationResult:
        response = self._client.chat(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            options={"temperature": 0.7, "num_predict": max_tokens},
        )
        return GenerationResult(
            text=response["message"]["content"].strip(),
            model=self._model,
            cost_usd=0.0,
        )


class OllamaGemmaProvider(LLMExtractionProvider):
    """
    Secondary text generator using Gemma3 via Ollama.

    Use for: captions, hooks, scripts (generation layer only)
    Do NOT use as primary extraction model — Qwen is better for structured output.

    Pull model: ollama pull gemma3:4b
    """
    def __init__(
        self,
        model: str = "gemma3:4b",
        base_url: str = "http://localhost:11434",
    ):
        import ollama
        self._client = ollama.Client(host=base_url)
        self._model = model

    def extract(self, transcript, ocr_text, caption, system_prompt, **kwargs) -> ExtractionResult:
        # Gemma can do extraction but Qwen is preferred — this is a fallback only
        return OllamaQwenVLProvider(model=self._model).extract(
            transcript, ocr_text, caption, system_prompt
        )

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 1000) -> GenerationResult:
        response = self._client.chat(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            options={"temperature": 0.7, "num_predict": max_tokens},
        )
        return GenerationResult(
            text=response["message"]["content"].strip(),
            model=self._model,
            cost_usd=0.0,
        )
```

---

### 4.5 `providers/embeddings.py`

```python
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Embed a single string. Returns a float list."""

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple strings. Default: call embed() per item."""
        return [self.embed(t) for t in texts]

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Number of dimensions in the embedding vector."""


class OllamaEmbeddingProvider(EmbeddingProvider):
    """
    Local embeddings via Ollama.

    Recommended models (pull before use):
      ollama pull qwen3-embedding    # 0.6B, 1024-dim, state-of-the-art retrieval
      ollama pull all-minilm         # tiny, 384-dim, very fast

    Note: if you switch embedding models after indexing some units, you MUST
    re-embed ALL existing units — mixed-dimension vectors cannot be compared.
    """
    _DIMS = {
        "qwen3-embedding": 1024,
        "all-minilm": 384,
        "nomic-embed-text": 768,
    }

    def __init__(
        self,
        model: str = "qwen3-embedding",
        base_url: str = "http://localhost:11434",
    ):
        import ollama
        self._client = ollama.Client(host=base_url)
        self._model = model
        self._dims = self._DIMS.get(model, 1024)

    def embed(self, text: str) -> List[float]:
        response = self._client.embeddings(model=self._model, prompt=text)
        return response["embedding"]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed(t) for t in texts]   # Ollama doesn't batch-embed yet

    @property
    def dimensions(self) -> int:
        return self._dims


class SentenceTransformerProvider(EmbeddingProvider):
    """
    Local embeddings via sentence-transformers (in-process, no server needed).

    Recommended model: all-MiniLM-L6-v2 (384-dim, ~80 MB, very fast on CPU)
    Higher quality: all-mpnet-base-v2 (768-dim, ~420 MB)

    Install: pip install sentence-transformers
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)
        self._dims = self._model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> List[float]:
        return self._model.encode(text).tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return self._model.encode(texts).tolist()

    @property
    def dimensions(self) -> int:
        return self._dims
```

---

### 4.6 `providers/factory.py`

```python
"""
Provider factory — reads provider config and returns the right implementation.

Config is read from .streamlit/secrets.toml:
  [providers]
  stt      = "faster-whisper"      # or "deepgram"
  llm      = "ollama-qwen"         # or "ollama-gemma" or "openai"
  embeddings = "ollama"            # or "sentence-transformers" or "openai"
  ocr      = "paddle"              # or "none"

Defaults to local providers if [providers] section is missing.
"""
from __future__ import annotations
from typing import Optional
import streamlit as st

from providers.stt import SpeechToTextProvider, FasterWhisperProvider
from providers.ocr import OCRProvider, PaddleOCRProvider
from providers.llm import LLMExtractionProvider, OllamaQwenVLProvider, OllamaGemmaProvider
from providers.embeddings import EmbeddingProvider, OllamaEmbeddingProvider, SentenceTransformerProvider


def _cfg(key: str, default: str) -> str:
    try:
        return st.secrets.get("providers", {}).get(key, default)
    except Exception:
        return default


def get_stt() -> SpeechToTextProvider:
    provider = _cfg("stt", "faster-whisper")
    if provider == "deepgram":
        # Keep old Deepgram path working during migration
        from processing.transcribe import DeepgramTranscriber
        # Wrap it so it conforms to the new interface
        return _DeepgramAdapter(st.secrets["DEEPGRAM_API_KEY"])
    return FasterWhisperProvider(model_size="large-v3", compute_type="int8")


def get_ocr() -> Optional[OCRProvider]:
    provider = _cfg("ocr", "paddle")
    if provider == "none":
        return None
    return PaddleOCRProvider()


def get_llm() -> LLMExtractionProvider:
    provider = _cfg("llm", "ollama-qwen")
    if provider == "openai":
        from providers.llm import _OpenAIAdapter
        return _OpenAIAdapter(st.secrets["OPENAI_API_KEY"])
    if provider == "ollama-gemma":
        return OllamaGemmaProvider()
    return OllamaQwenVLProvider()


def get_embeddings() -> EmbeddingProvider:
    provider = _cfg("embeddings", "sentence-transformers")
    if provider == "openai":
        from providers.embeddings import _OpenAIEmbeddingAdapter
        return _OpenAIEmbeddingAdapter(st.secrets["OPENAI_API_KEY"])
    if provider == "ollama":
        return OllamaEmbeddingProvider()
    return SentenceTransformerProvider()   # safest default: no server needed
```

---

## 5. DB dimension migration

⚠️ **Critical:** The current DB stores embeddings as `FLOAT[1536]` (OpenAI dimensions).  
Local models have different dimensions:
- `qwen3-embedding`: 1024-dim
- `all-MiniLM-L6-v2`: 384-dim

When you switch embedding providers, you must:
1. Drop the existing `embedding` column
2. Re-create it with the new dimension type
3. Re-embed all existing message units

Migration SQL (run once when switching):
```sql
-- Step 1: drop old embeddings
UPDATE message_units SET embedding = NULL, embedded_at = NULL;
-- Step 2: alter column type (DuckDB requires recreate)
ALTER TABLE message_units DROP COLUMN embedding;
ALTER TABLE message_units ADD COLUMN embedding FLOAT[1024];   -- qwen3-embedding
-- Step 3: re-run embedding pipeline via the UI
```

Add this migration to `core/db.py`'s `init_db()` with a version flag to avoid running twice.

---

## 6. Migration tasks (ordered by effort)

### Task 1 — Add faster-whisper STT provider (est. 2 hours)

**Goal:** Replace Deepgram with faster-whisper. Zero API cost from this point.

Steps:
1. `pip install faster-whisper`
2. Create `providers/__init__.py`, `providers/stt.py` (copy interface from Section 4.2)
3. Modify `processing/transcribe.py`:
   - Import `FasterWhisperProvider` from `providers.stt`
   - Make `transcribe_post()` accept an optional `SpeechToTextProvider` parameter
   - Default to `FasterWhisperProvider()` when no provider is passed
   - Keep `DeepgramTranscriber` in the file as the fallback path for users who still have a key
4. Update `pages/3_📝_Corpus_Explorer.py`:
   - Remove the Deepgram API key input field
   - Add a note: "Using local faster-whisper (no API key needed)"
5. Test with one post: `export PYTHONPATH=. && venv/bin/python -c "from providers.stt import FasterWhisperProvider; p = FasterWhisperProvider(); print(p.transcribe('audio_extracts/test.wav', 'test'))"` 

**First run downloads model weights (~1.5 GB) to `~/.cache/huggingface/`.**

---

### Task 2 — Add PaddleOCR frame extraction (est. 3 hours)

**Goal:** Extract on-screen text from video frames (new capability — zero API cost).

Steps:
1. `pip install paddleocr paddlepaddle`
2. Create `providers/ocr.py` (copy from Section 4.3)
3. Create `processing/frames.py`:
   ```python
   # Use ffmpeg to sample N frames from a video at evenly-spaced intervals
   # Save as JPEG to frames/{post_id}/frame_{n:03d}.jpg
   # Return list of frame paths
   def sample_frames(video_path: str, n_frames: int = 6) -> list[str]: ...
   ```
4. Create `processing/ocr_queue.py`:
   - Find posts with `download_status = 'done'` and no OCR data yet
   - Call `sample_frames()` then `PaddleOCRProvider().extract_text()` per frame
   - Merge all frame text into a single string, store in new `posts.ocr_text` column
5. Add `ocr_text TEXT` column to `posts` table in `core/db.py` (with ALTER TABLE migration)
6. Update extraction pipeline to pass `ocr_text` as input alongside transcript

---

### Task 3 — Add Ollama + Qwen VL extraction provider (est. 3 hours)

**Goal:** Replace GPT-4o-mini extraction. Zero API cost from this point.

Prerequisites: `brew install ollama && ollama pull qwen2.5vl:7b`

Steps:
1. `pip install ollama`
2. Create `providers/llm.py` (copy from Section 4.4)
3. Modify `analysis/extraction.py`:
   - Import `LLMExtractionProvider` from `providers.llm`
   - Make `extract_message_units()` accept an optional `LLMExtractionProvider` parameter
   - Default to `OllamaQwenVLProvider()` when no provider passed
   - Keep existing OpenAI HTTP call as the fallback (`provider='openai'`)
   - Add `ocr_text: str = ""` parameter to pass frame text into the prompt
4. Modify the extraction system prompt in `analysis/extraction.py`:
   - Add a new section: `"On-screen text (from video frames): {ocr_text}"` before the transcript
   - Instruct the model to treat OCR text as supplementary evidence
5. Update `processing/extraction_queue.py`:
   - Look up `ocr_text` from `posts` table and pass it to `extract_message_units()`

---

### Task 4 — Add local embeddings (est. 1.5 hours)

**Goal:** Replace OpenAI embeddings. Zero API cost from this point.

**⚠️ Breaking change:** switching dimension requires re-embedding everything.

Steps:
1. Choose embedding model: `all-MiniLM-L6-v2` (384-dim, fastest) OR `qwen3-embedding` (1024-dim, best quality)
2. `pip install sentence-transformers` (or `ollama pull qwen3-embedding`)
3. Create `providers/embeddings.py` (copy from Section 4.5)
4. Run the DB migration in Section 5 to resize the embedding column
5. Update `analysis/embeddings.py`:
   - Import `EmbeddingProvider` from `providers.embeddings`
   - Make `embed_pending_units()` accept an optional `EmbeddingProvider` parameter
   - Default to `SentenceTransformerProvider()` or `OllamaEmbeddingProvider()`
6. Run `embed_pending_units()` to re-embed all existing units with the new provider
7. Update `analysis/search.py`:
   - Change the DuckDB cast in `semantic_search()` from `FLOAT[1536]` to match the new dimension
   - e.g.: `?::FLOAT[384]` for MiniLM, `?::FLOAT[1024]` for qwen3-embedding

---

### Task 5 — Add Ollama Gemma generation (est. 1.5 hours)

**Goal:** Replace GPT-4o-mini in Content Studio. Zero API cost from this point.

Prerequisite: `ollama pull gemma3:4b`

Steps:
1. Update `providers/llm.py` with `OllamaGemmaProvider` (Section 4.4)
2. Modify `analysis/content_gen.py`:
   - Import `LLMExtractionProvider` from `providers.llm`
   - Make `_call_gpt()` accept an optional provider parameter
   - When provider is an `OllamaGemmaProvider`, call `provider.generate()` instead of requests.post
3. Update `pages/6_✍️_Content_Studio.py`:
   - Remove the `OPENAI_API_KEY` hard-stop at the top
   - Add a provider selector if needed (Gemma local vs OpenAI cloud)

---

### Task 6 — Provider factory + secrets config (est. 1 hour)

**Goal:** Let the user switch providers without touching code.

Steps:
1. Create `providers/factory.py` (copy from Section 4.6)
2. Add to `.streamlit/secrets.toml`:
   ```toml
   [providers]
   stt        = "faster-whisper"     # change to "deepgram" to revert
   llm        = "ollama-qwen"        # change to "openai" to revert
   embeddings = "sentence-transformers"  # change to "openai" to revert
   ocr        = "paddle"             # change to "none" to disable
   ```
3. Update all queue runners to call `factory.get_stt()` etc. instead of importing providers directly

---

## 7. Testing plan for each task

Each task has a quick smoke test — run before committing.

```bash
# Task 1 — STT
export PYTHONPATH=/Users/ariankalantari/senpai-reel
venv/bin/python -c "
from providers.stt import FasterWhisperProvider
p = FasterWhisperProvider(model_size='small')   # use small for quick test
r = p.transcribe('audio_extracts/YOUR_FILE.wav', 'test-post')
print(r.transcript[:200])
print(f'{r.duration_sec:.1f}s, {r.word_count} words')
"

# Task 2 — OCR
venv/bin/python -c "
from providers.ocr import PaddleOCRProvider
p = PaddleOCRProvider()
r = p.extract_text('downloads/YOUR_FRAME.jpg')
print(r.full_text)
"

# Task 3 — LLM (requires Ollama running: ollama serve)
venv/bin/python -c "
from providers.llm import OllamaQwenVLProvider
p = OllamaQwenVLProvider(model='qwen2.5vl:3b')
r = p.extract('Your resume should have keywords.', '', '', 'Extract tips as JSON {units:[]}.')
print(r.units)
"

# Task 4 — Embeddings
venv/bin/python -c "
from providers.embeddings import SentenceTransformerProvider
p = SentenceTransformerProvider()
v = p.embed('how to write a resume for ATS')
print(f'{len(v)} dims, first 5: {v[:5]}')
"
```

---

## 8. Rollback strategy

Each task keeps the old provider code in place and adds the new provider alongside it.  
Switching back is always a single line in `secrets.toml` (e.g. `stt = "deepgram"`).

Never delete the old provider code until the new one has been running in production for at least 2 weeks.

---

## 9. Recommended implementation order

| Priority | Task | Benefit | API cost saved/month |
|----------|------|---------|---------------------|
| 1 | Task 1 — faster-whisper | Biggest recurring cost eliminated | Deepgram ~$58 for 10k min |
| 2 | Task 4 — local embeddings | Second biggest cost, also unblocks search | OpenAI ~$2 per 100M tokens |
| 3 | Task 2 — PaddleOCR | New capability (fills OCR gap) | — |
| 4 | Task 3 — Qwen VL extraction | Replaces GPT-4o-mini extraction | OpenAI ~$0.30 per 1k reels |
| 5 | Task 5 — Gemma generation | Replaces Content Studio GPT | OpenAI ~$1 per 1k generations |
| 6 | Task 6 — factory config | Ties it all together cleanly | — |

Start with Task 1 (fastest to test, biggest cost impact) and Task 4 (can run in parallel — no Ollama needed for SentenceTransformers).
