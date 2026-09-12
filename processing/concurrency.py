"""Bounded concurrency helpers for pipeline queues."""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, TypeVar


DEFAULT_IO_WORKERS = 8
DEFAULT_CPU_WORKERS = 2

_DB_WRITE_LOCK = threading.Lock()

ItemT = TypeVar("ItemT")
ResultT = TypeVar("ResultT")


@dataclass(frozen=True)
class CompletedWork:
    """Result for one concurrently processed item."""

    index: int
    item: object
    result: object | None = None
    error: Exception | None = None


def _coerce_worker_count(value: object, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(parsed, 1)


def _worker_count(explicit: Optional[int], env_name: str, default: int) -> int:
    if explicit is not None:
        return _coerce_worker_count(explicit, default)
    return _coerce_worker_count(os.getenv(env_name), default)


def io_worker_count(explicit: Optional[int] = None) -> int:
    """Workers for I/O-bound CDN/API calls.

    Defaults to 8: enough to hide network latency without hammering APIs or a
    fanless laptop. Override with SENPAI_IO_WORKERS or a function argument.
    """

    return _worker_count(explicit, "SENPAI_IO_WORKERS", DEFAULT_IO_WORKERS)


def cpu_worker_count(explicit: Optional[int] = None) -> int:
    """Workers for CPU-heavy local work such as ffmpeg."""

    return _worker_count(explicit, "SENPAI_CPU_WORKERS", DEFAULT_CPU_WORKERS)


@contextmanager
def duckdb_write_lock():
    """Serialise DuckDB writes while allowing parallel off-thread work."""

    with _DB_WRITE_LOCK:
        yield


def run_db_write(func: Callable[..., ResultT], *args, **kwargs) -> ResultT:
    """Run a write function behind the shared DuckDB writer lock."""

    with duckdb_write_lock():
        return func(*args, **kwargs)


def run_bounded(
    items: Iterable[ItemT],
    worker: Callable[[ItemT], ResultT],
    max_workers: int,
    progress_callback: Optional[Callable[[int, int, ItemT, ResultT | None, Exception | None], None]] = None,
) -> list[CompletedWork]:
    """Run item work with bounded threads and ordered progress counts.

    Worker failures are captured per item instead of cancelling the whole run.
    Progress callbacks are emitted by the coordinating thread as futures finish,
    so Streamlit/heartbeat callers do not receive concurrent callbacks.
    """

    work_items = list(items)
    total = len(work_items)
    if total == 0:
        return []

    worker_limit = min(_coerce_worker_count(max_workers, 1), total)
    completed: list[CompletedWork] = []

    with ThreadPoolExecutor(max_workers=worker_limit) as executor:
        futures = {
            executor.submit(worker, item): (index, item)
            for index, item in enumerate(work_items)
        }
        done_count = 0
        for future in as_completed(futures):
            index, item = futures[future]
            result = None
            error = None
            try:
                result = future.result()
            except Exception as exc:
                error = exc

            done_count += 1
            if progress_callback:
                progress_callback(done_count, total, item, result, error)
            completed.append(CompletedWork(index=index, item=item, result=result, error=error))

    return sorted(completed, key=lambda item: item.index)
