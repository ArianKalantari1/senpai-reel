import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import duckdb
import pytest


@pytest.fixture
def media_db(tmp_path, monkeypatch):
    import core.db as db_mod

    old_path = db_mod.DB_PATH
    db_mod.DB_PATH = str(tmp_path / "media.duckdb")
    db_mod.init_db()
    yield db_mod
    db_mod.DB_PATH = old_path


def _seed_ready_post(
    db_mod,
    tmp_path,
    post_id="post_media",
    with_video=True,
    with_transcript=True,
    with_keyframes=True,
):
    client_id = db_mod.DEFAULT_CLIENT_ID
    now = datetime.utcnow()

    downloads = tmp_path / "downloads"
    downloads.mkdir(exist_ok=True)
    video_path = downloads / f"{post_id}.mp4"
    if with_video:
        video_path.write_bytes(b"video")

    audio_path = tmp_path / "audio" / f"{post_id}.wav"
    audio_path.parent.mkdir(exist_ok=True)
    audio_path.write_bytes(b"RIFF audio")

    keyframes_dir = None
    keyframe_count = None
    if with_keyframes:
        keyframes = tmp_path / "keyframes" / post_id
        keyframes.mkdir(parents=True)
        (keyframes / "frame_001.jpg").write_bytes(b"jpg")
        keyframes_dir = str(keyframes)
        keyframe_count = 1

    conn = duckdb.connect(db_mod.DB_PATH)
    try:
        conn.execute(
            """
            INSERT INTO posts (
                post_id, client_id, account_id, engagement_rate, download_status,
                scraped_at, hashtags, mentions, local_video_path, local_audio_path,
                keyframes_dir, keyframe_count
            )
            VALUES (?, ?, 'acc1', 0, 'done', ?, [], [], ?, ?, ?, ?)
            """,
            [
                post_id,
                client_id,
                now,
                str(video_path) if with_video else str(video_path),
                str(audio_path),
                keyframes_dir,
                keyframe_count,
            ],
        )
        conn.execute(
            "INSERT INTO client_posts (client_id, post_id, added_at) VALUES (?, ?, ?)",
            [client_id, post_id, now],
        )
        if with_transcript:
            conn.execute(
                """
                INSERT INTO transcripts (
                    post_id, client_id, provider, model, transcript, language,
                    confidence, duration_sec, word_count, transcribed_at, cost_usd
                )
                VALUES (?, ?, 'deepgram', 'nova-2', 'hello', 'en', 0.9, 5, 1, ?, 0.001)
                """,
                [post_id, client_id, now],
            )
    finally:
        conn.close()

    return {
        "client_id": client_id,
        "video_path": video_path,
        "audio_path": audio_path,
        "keyframes_dir": keyframes_dir,
    }


def test_archive_unset_leaves_video_in_place(media_db, tmp_path, monkeypatch):
    from processing.media_archive import archive_video_if_ready

    seeded = _seed_ready_post(media_db, tmp_path, "unset")
    monkeypatch.delenv("SENPAI_ARCHIVE_DIR", raising=False)

    result = archive_video_if_ready("unset", client_id=seeded["client_id"])

    assert result["status"] == "skipped"
    assert result["reason"] == "archive_unset"
    assert seeded["video_path"].exists()


def test_archive_target_disconnected_is_a_skip_not_archive(media_db, tmp_path):
    from processing.media_archive import archive_video_if_ready

    seeded = _seed_ready_post(media_db, tmp_path, "disconnected")
    missing_archive = tmp_path / "external_ssd_not_mounted"

    result = archive_video_if_ready(
        "disconnected",
        client_id=seeded["client_id"],
        archive_dir=str(missing_archive),
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "archive_unreachable"
    assert seeded["video_path"].exists()

    conn = duckdb.connect(media_db.DB_PATH)
    row = conn.execute(
        "SELECT local_video_path, archived_video_path FROM posts WHERE post_id = 'disconnected'"
    ).fetchone()
    conn.close()
    assert row[0] == str(seeded["video_path"])
    assert row[1] is None


def test_archive_moves_video_and_records_location(media_db, tmp_path):
    from processing.media_archive import archive_video_if_ready

    seeded = _seed_ready_post(media_db, tmp_path, "archive_me")
    archive_root = tmp_path / "external_archive"
    archive_root.mkdir()

    result = archive_video_if_ready(
        "archive_me",
        client_id=seeded["client_id"],
        archive_dir=str(archive_root),
    )

    assert result["status"] == "archived"
    assert not seeded["video_path"].exists()
    archived_path = Path(result["path"])
    assert archived_path.exists()
    assert archived_path.read_bytes() == b"video"

    conn = duckdb.connect(media_db.DB_PATH)
    row = conn.execute(
        """
        SELECT local_video_path, archived_video_path, video_archived_at
        FROM posts
        WHERE post_id = 'archive_me'
        """
    ).fetchone()
    conn.close()
    assert row[0] is None
    assert row[1] == str(archived_path)
    assert row[2] is not None


def test_missing_source_is_not_marked_archived(media_db, tmp_path):
    from processing.media_archive import archive_video_if_ready

    seeded = _seed_ready_post(media_db, tmp_path, "missing_source", with_video=False)
    archive_root = tmp_path / "external_archive"
    archive_root.mkdir()

    result = archive_video_if_ready(
        "missing_source",
        client_id=seeded["client_id"],
        archive_dir=str(archive_root),
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "source_file_missing"

    conn = duckdb.connect(media_db.DB_PATH)
    archived = conn.execute(
        "SELECT archived_video_path FROM posts WHERE post_id = 'missing_source'"
    ).fetchone()[0]
    conn.close()
    assert archived is None


def test_audio_extraction_does_not_repeat_after_video_moved(media_db, tmp_path):
    from processing.audio import extract_audio_for_post

    seeded = _seed_ready_post(media_db, tmp_path, "moved_already")
    seeded["video_path"].unlink()

    conn = duckdb.connect(media_db.DB_PATH)
    conn.execute(
        """
        UPDATE posts
        SET local_video_path = NULL, archived_video_path = ?
        WHERE post_id = 'moved_already'
        """,
        [str(tmp_path / "archive" / "moved_already.mp4")],
    )
    conn.close()

    result = extract_audio_for_post("moved_already", client_id=seeded["client_id"])

    assert result["success"] is True
    assert result["path"] == str(seeded["audio_path"])


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_extract_keyframes_from_real_video(tmp_path):
    from processing.media_archive import extract_keyframes

    video = tmp_path / "sample.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=320x480:r=24",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=44100",
            "-t",
            "4",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-c:a",
            "aac",
            str(video),
        ],
        check=True,
    )

    result = extract_keyframes(str(video), "real_video", output_dir=str(tmp_path / "frames"))

    assert result["success"] is True
    assert result["count"] > 0
    assert Path(result["path"]).exists()
