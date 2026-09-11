import threading
import time


def test_run_bounded_respects_worker_limit():
    from processing.concurrency import run_bounded

    active = 0
    max_active = 0
    lock = threading.Lock()

    def worker(item):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return item * 2

    results = run_bounded(range(12), worker, max_workers=3)

    assert max_active <= 3
    assert [item.result for item in results] == [i * 2 for i in range(12)]


def test_run_bounded_keeps_progress_callbacks_serial():
    from processing.concurrency import run_bounded

    active_callbacks = 0
    max_active_callbacks = 0
    counts = []
    lock = threading.Lock()

    def worker(item):
        time.sleep(0.01)
        return item

    def progress(done, total, item, result, error):
        nonlocal active_callbacks, max_active_callbacks
        with lock:
            active_callbacks += 1
            max_active_callbacks = max(max_active_callbacks, active_callbacks)
            counts.append(done)
        time.sleep(0.01)
        with lock:
            active_callbacks -= 1

    run_bounded(range(8), worker, max_workers=4, progress_callback=progress)

    assert counts == list(range(1, 9))
    assert max_active_callbacks == 1


def test_worker_counts_are_configurable(monkeypatch):
    from processing.concurrency import cpu_worker_count, io_worker_count

    monkeypatch.setenv("SENPAI_IO_WORKERS", "11")
    monkeypatch.setenv("SENPAI_CPU_WORKERS", "3")

    assert io_worker_count() == 11
    assert cpu_worker_count() == 3
    assert io_worker_count(5) == 5
    assert cpu_worker_count(1) == 1
