"""Benchmark bounded pipeline concurrency without spending API credits.

This measures two representative bottlenecks:
- I/O-style per-item work with a serialized DuckDB-like write section.
- Actual ffmpeg audio extraction from generated local mp4 files.

It intentionally avoids live CDN, Deepgram, and OpenAI calls.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from processing.audio import extract_audio
from processing.concurrency import cpu_worker_count, io_worker_count, run_bounded, run_db_write


def _measure(func):
    started = time.perf_counter()
    func()
    return round(time.perf_counter() - started, 3)


def _speedup(serial_sec: float, parallel_sec: float) -> float:
    if parallel_sec <= 0:
        return 0.0
    return round(serial_sec / parallel_sec, 2)


def _bench_io(items: int, workers: int, item_delay: float, write_delay: float) -> float:
    def _work(_item):
        time.sleep(item_delay)
        run_db_write(time.sleep, write_delay)
        return True

    return _measure(lambda: run_bounded(range(items), _work, max_workers=workers))


def _make_sample_video(path: Path, seconds: int):
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=540x960:r=30",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:sample_rate=44100",
        "-t",
        str(seconds),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-c:a",
        "aac",
        str(path),
    ]
    subprocess.run(cmd, check=True)


def _bench_audio(items: int, workers: int, seconds: int, tmp_dir: Path) -> float | None:
    if shutil.which("ffmpeg") is None:
        return None

    sample = tmp_dir / "sample.mp4"
    _make_sample_video(sample, seconds)
    videos_dir = tmp_dir / f"videos_{workers}"
    audio_dir = tmp_dir / f"audio_{workers}"
    videos_dir.mkdir()
    audio_dir.mkdir()
    video_paths = []
    for index in range(items):
        path = videos_dir / f"bench_{index}.mp4"
        shutil.copyfile(sample, path)
        video_paths.append(path)

    def _work(path: Path):
        result = extract_audio(str(path), output_dir=str(audio_dir))
        if not result["success"]:
            raise RuntimeError(result["error"])
        return result

    return _measure(lambda: run_bounded(video_paths, _work, max_workers=workers))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=int, default=96)
    parser.add_argument("--io-workers", type=int, default=io_worker_count())
    parser.add_argument("--cpu-workers", type=int, default=cpu_worker_count())
    parser.add_argument("--io-delay", type=float, default=0.08)
    parser.add_argument("--write-delay", type=float, default=0.003)
    parser.add_argument("--audio-items", type=int, default=24)
    parser.add_argument("--audio-seconds", type=int, default=12)
    args = parser.parse_args()

    results = {
        "io_simulation": {
            "items": args.items,
            "item_delay_sec": args.io_delay,
            "serialized_write_delay_sec": args.write_delay,
            "serial_workers": 1,
            "parallel_workers": args.io_workers,
        },
        "audio_ffmpeg": {
            "items": args.audio_items,
            "seconds_per_video": args.audio_seconds,
            "serial_workers": 1,
            "parallel_workers": args.cpu_workers,
        },
    }

    io_serial = _bench_io(args.items, 1, args.io_delay, args.write_delay)
    io_parallel = _bench_io(args.items, args.io_workers, args.io_delay, args.write_delay)
    results["io_simulation"].update(
        {
            "serial_sec": io_serial,
            "parallel_sec": io_parallel,
            "speedup": _speedup(io_serial, io_parallel),
        }
    )

    with tempfile.TemporaryDirectory(prefix="senpai-concurrency-bench-") as tmp:
        tmp_dir = Path(tmp)
        audio_serial = _bench_audio(args.audio_items, 1, args.audio_seconds, tmp_dir)
        audio_parallel = _bench_audio(
            args.audio_items,
            args.cpu_workers,
            args.audio_seconds,
            tmp_dir,
        )

    if audio_serial is None or audio_parallel is None:
        results["audio_ffmpeg"]["skipped"] = "ffmpeg not found"
    else:
        results["audio_ffmpeg"].update(
            {
                "serial_sec": audio_serial,
                "parallel_sec": audio_parallel,
                "speedup": _speedup(audio_serial, audio_parallel),
            }
        )

    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
