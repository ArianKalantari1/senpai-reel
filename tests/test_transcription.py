"""
Tests for processing/transcribe.py — mocked Deepgram API + transcript parsing.
"""
import json
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path


# ── DeepgramTranscriber.__init__ ─────────────────────────────────────────────

class TestDeepgramTranscriberInit:
    def test_valid_key_initialises(self):
        from processing.transcribe import DeepgramTranscriber
        t = DeepgramTranscriber("my_api_key")
        assert "Token my_api_key" in t._headers["Authorization"]

    def test_empty_key_raises(self):
        from processing.transcribe import DeepgramTranscriber
        with pytest.raises(ValueError):
            DeepgramTranscriber("")

    def test_none_key_raises(self):
        from processing.transcribe import DeepgramTranscriber
        with pytest.raises((ValueError, TypeError)):
            DeepgramTranscriber(None)


# ── DeepgramTranscriber.transcribe ───────────────────────────────────────────

def _deepgram_response_json(
    transcript="Hello world",
    confidence=0.99,
    duration=5.0,
    words=None,
):
    """Build a minimal Deepgram API response dict."""
    if words is None:
        words = [
            {"word": "Hello", "start": 0.0, "end": 0.5, "confidence": 0.99},
            {"word": "world", "start": 0.6, "end": 1.0, "confidence": 0.98},
        ]
    return {
        "results": {
            "channels": [
                {
                    "alternatives": [
                        {
                            "transcript": transcript,
                            "confidence": confidence,
                            "words": words,
                        }
                    ]
                }
            ]
        },
        "metadata": {"duration": duration},
    }


class TestDeepgramTranscribe:
    def test_missing_file_raises(self, tmp_path):
        from processing.transcribe import DeepgramTranscriber
        t = DeepgramTranscriber("key")
        with pytest.raises(FileNotFoundError):
            t.transcribe(str(tmp_path / "missing.wav"), "post_001")

    def test_successful_transcription(self, tmp_path):
        from processing.transcribe import DeepgramTranscriber, TranscriptResult

        # Create a dummy WAV file
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = _deepgram_response_json(
            transcript="Get your resume right",
            duration=10.0,
        )

        with patch("processing.transcribe.requests.post", return_value=mock_resp):
            t = DeepgramTranscriber("valid_key")
            result = t.transcribe(str(audio_file), "post_001")

        assert isinstance(result, TranscriptResult)
        assert result.transcript == "Get your resume right"
        assert result.provider == "deepgram"
        assert result.model == "nova-2"
        assert result.duration_sec == pytest.approx(10.0, abs=0.1)
        assert result.post_id == "post_001"

    def test_401_raises_permission_error(self, tmp_path):
        from processing.transcribe import DeepgramTranscriber

        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"fake")

        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.raise_for_status.return_value = None

        with patch("processing.transcribe.requests.post", return_value=mock_resp):
            t = DeepgramTranscriber("bad_key")
            with pytest.raises(PermissionError):
                t.transcribe(str(audio_file), "post_001")

    def test_word_timestamps_parsed(self, tmp_path):
        from processing.transcribe import DeepgramTranscriber

        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"fake")

        words = [
            {"word": "resume", "start": 0.0, "end": 0.3, "confidence": 0.95},
            {"word": "tips", "start": 0.4, "end": 0.7, "confidence": 0.90},
        ]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = _deepgram_response_json(words=words)

        with patch("processing.transcribe.requests.post", return_value=mock_resp):
            t = DeepgramTranscriber("key")
            result = t.transcribe(str(audio_file), "post_002")

        assert len(result.words) == 2
        assert result.words[0].word == "resume"
        assert result.words[1].start_sec == pytest.approx(0.4)

    def test_cost_calculated_from_duration(self, tmp_path):
        from processing.transcribe import DeepgramTranscriber, _DEEPGRAM_COST_PER_MIN

        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"fake")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = _deepgram_response_json(duration=60.0)

        with patch("processing.transcribe.requests.post", return_value=mock_resp):
            t = DeepgramTranscriber("key")
            result = t.transcribe(str(audio_file), "post_003")

        expected_cost = (60.0 / 60) * _DEEPGRAM_COST_PER_MIN
        assert result.cost_usd == pytest.approx(expected_cost, rel=0.01)


# ── TranscriptResult.word_count ───────────────────────────────────────────────

class TestTranscriptResult:
    def test_word_count_property(self):
        from processing.transcribe import TranscriptResult
        r = TranscriptResult(
            post_id="p1",
            provider="deepgram",
            model="nova-2",
            transcript="one two three four five",
            language="en",
            confidence=0.95,
            duration_sec=10.0,
        )
        assert r.word_count == 5

    def test_word_count_empty_transcript(self):
        from processing.transcribe import TranscriptResult
        r = TranscriptResult(
            post_id="p2",
            provider="deepgram",
            model="nova-2",
            transcript="",
            language="en",
            confidence=0.0,
            duration_sec=0.0,
        )
        assert r.word_count == 0


# ── WhisperTranscriber ────────────────────────────────────────────────────────

class TestWhisperTranscriber:
    def test_empty_key_raises(self):
        from processing.transcribe import WhisperTranscriber
        with pytest.raises(ValueError):
            WhisperTranscriber("")

    def test_missing_file_raises(self, tmp_path):
        from processing.transcribe import WhisperTranscriber
        t = WhisperTranscriber("key")
        with pytest.raises(FileNotFoundError):
            t.transcribe(str(tmp_path / "missing.wav"), "p1")

    def test_successful_transcription(self, tmp_path):
        from processing.transcribe import WhisperTranscriber, TranscriptResult

        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"\x00" * 32000)  # 1 second at 16kHz 16-bit

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"text": "Resume advice here"}

        with patch("processing.transcribe.requests.post", return_value=mock_resp):
            t = WhisperTranscriber("key")
            result = t.transcribe(str(audio_file), "p_whisper")

        assert isinstance(result, TranscriptResult)
        assert result.provider == "openai"
        assert result.model == "whisper-1"
        assert result.transcript == "Resume advice here"
        assert result.words == []  # Whisper doesn't return word timestamps


# ── save_transcript ───────────────────────────────────────────────────────────

class TestSaveTranscript:
    def test_saves_to_db(self, tmp_path):
        import duckdb
        import core.db as db_mod
        old_path = db_mod.DB_PATH
        db_mod.DB_PATH = str(tmp_path / "save_t.duckdb")
        db_mod.init_db()

        from processing.transcribe import (
            TranscriptResult, WordTimestamp, save_transcript
        )

        result = TranscriptResult(
            post_id="post_save",
            provider="deepgram",
            model="nova-2",
            transcript="Hello world",
            language="en",
            confidence=0.95,
            duration_sec=5.0,
            words=[WordTimestamp("Hello", 0.0, 0.5, 0.99)],
            cost_usd=0.001,
        )
        save_transcript(result)

        conn = duckdb.connect(db_mod.DB_PATH)
        row = conn.execute(
            "SELECT transcript FROM transcripts WHERE post_id = 'post_save'"
        ).fetchone()
        word_count = conn.execute(
            "SELECT COUNT(*) FROM transcript_words WHERE post_id = 'post_save'"
        ).fetchone()[0]
        conn.close()

        assert row[0] == "Hello world"
        assert word_count == 1

        db_mod.DB_PATH = old_path

    def test_upsert_replaces_existing(self, tmp_path):
        import duckdb
        import core.db as db_mod
        old_path = db_mod.DB_PATH
        db_mod.DB_PATH = str(tmp_path / "upsert_t.duckdb")
        db_mod.init_db()

        from processing.transcribe import TranscriptResult, save_transcript

        def _make(text):
            return TranscriptResult("post_upsert", "deepgram", "nova-2", text, "en", 0.9, 5.0)

        save_transcript(_make("first version"))
        save_transcript(_make("second version"))

        conn = duckdb.connect(db_mod.DB_PATH)
        count = conn.execute(
            "SELECT COUNT(*) FROM transcripts WHERE post_id = 'post_upsert'"
        ).fetchone()[0]
        text = conn.execute(
            "SELECT transcript FROM transcripts WHERE post_id = 'post_upsert'"
        ).fetchone()[0]
        conn.close()

        assert count == 1
        assert text == "second version"

        db_mod.DB_PATH = old_path


# ── transcribe_post ───────────────────────────────────────────────────────────

class TestTranscribePost:
    def test_raises_if_no_audio_path(self, tmp_path):
        import core.db as db_mod
        old_path = db_mod.DB_PATH
        db_mod.DB_PATH = str(tmp_path / "tp.duckdb")
        db_mod.init_db()

        from processing.transcribe import transcribe_post
        # Post doesn't exist → should raise
        with pytest.raises(ValueError, match="No audio file"):
            transcribe_post("nonexistent_post", "key", "deepgram")

        db_mod.DB_PATH = old_path

    def test_transcribes_and_saves(self, tmp_path):
        import duckdb
        import core.db as db_mod
        old_path = db_mod.DB_PATH
        db_mod.DB_PATH = str(tmp_path / "tp2.duckdb")
        db_mod.init_db()

        # Create audio file
        audio_file = tmp_path / "audio.wav"
        audio_file.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")

        # Insert post with local_audio_path
        conn = duckdb.connect(db_mod.DB_PATH)
        from datetime import datetime
        now = datetime.utcnow()
        client_id = db_mod.DEFAULT_CLIENT_ID
        conn.execute("""
            INSERT INTO posts (post_id, client_id, account_id, engagement_rate, download_status,
                scraped_at, hashtags, mentions, local_audio_path)
            VALUES ('post_tp', ?, 'acc1', 0, 'done', ?, [], [], ?)
        """, (client_id, now, str(audio_file)))
        conn.execute(
            "INSERT INTO client_posts (client_id, post_id, added_at) VALUES (?, 'post_tp', ?)",
            [client_id, now],
        )
        conn.close()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = _deepgram_response_json("Interview tips here", duration=5.0)

        with patch("processing.transcribe.requests.post", return_value=mock_resp):
            from processing.transcribe import transcribe_post
            result = transcribe_post("post_tp", "key", "deepgram")

        assert result.transcript == "Interview tips here"

        conn = duckdb.connect(db_mod.DB_PATH)
        saved = conn.execute(
            "SELECT transcript FROM transcripts WHERE post_id = 'post_tp'"
        ).fetchone()
        conn.close()
        assert saved[0] == "Interview tips here"

        db_mod.DB_PATH = old_path
