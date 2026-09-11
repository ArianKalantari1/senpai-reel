"""Lock recovery when the owning process dies (creative-director-ai #19)."""
from datetime import timedelta

import pytest

import core.pipeline as pl


@pytest.fixture
def clean_lock(tmp_path, monkeypatch):
    import core.db as db
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "t.duckdb"))
    monkeypatch.setattr(pl, "DB_PATH", str(tmp_path / "t.duckdb"), raising=False)
    db.init_db()
    yield


def _age_heartbeat(run_id: str, delta: timedelta):
    """Simulate a process that stopped sending heartbeats."""
    import core.db as db
    conn = db.get_connection()
    try:
        conn.execute(
            "UPDATE pipeline_locks SET heartbeat_at = heartbeat_at - INTERVAL (?) SECOND "
            "WHERE run_id = ?",
            [int(delta.total_seconds()), run_id],
        )
    finally:
        conn.close()


class TestForceRelease:
    def test_fresh_lock_cannot_be_force_released(self, clean_lock):
        run = pl.try_start_pipeline_run("c1")
        assert run["acquired"]
        out = pl.force_release_pipeline_lock(run["run_id"])
        assert out["released"] is False
        assert "still alive" in out["reason"]
        assert pl.get_pipeline_lock() is not None, "a live lock must survive"

    def test_abandoned_lock_can_be_released(self, clean_lock):
        run = pl.try_start_pipeline_run("c1")
        _age_heartbeat(run["run_id"], pl.LOCK_ABANDONED_AFTER + timedelta(minutes=1))
        assert pl.get_pipeline_lock()["abandoned"] is True
        out = pl.force_release_pipeline_lock(run["run_id"])
        assert out["released"] is True
        assert pl.get_pipeline_lock() is None

    def test_stale_run_id_is_refused(self, clean_lock):
        run = pl.try_start_pipeline_run("c1")
        _age_heartbeat(run["run_id"], pl.LOCK_ABANDONED_AFTER + timedelta(minutes=1))
        out = pl.force_release_pipeline_lock("some_other_run")
        assert out["released"] is False
        assert "changed since" in out["reason"]
        assert pl.get_pipeline_lock() is not None

    def test_release_with_no_lock_is_safe(self, clean_lock):
        out = pl.force_release_pipeline_lock("anything")
        assert out["released"] is False
        assert "No pipeline lock" in out["reason"]

    def test_heartbeat_refresh_clears_abandoned(self, clean_lock):
        run = pl.try_start_pipeline_run("c1")
        _age_heartbeat(run["run_id"], pl.LOCK_ABANDONED_AFTER + timedelta(minutes=1))
        assert pl.get_pipeline_lock()["abandoned"] is True
        pl.update_pipeline_lock(run["run_id"], "Transcribing")
        lock = pl.get_pipeline_lock()
        assert lock["abandoned"] is False, "a live run must not look abandoned"
        assert lock["stage"] == "Transcribing"

    def test_released_lock_frees_the_pipeline(self, clean_lock):
        first = pl.try_start_pipeline_run("c1")
        blocked = pl.try_start_pipeline_run("c2")
        assert blocked["acquired"] is False
        _age_heartbeat(first["run_id"], pl.LOCK_ABANDONED_AFTER + timedelta(minutes=1))
        pl.force_release_pipeline_lock(first["run_id"])
        assert pl.try_start_pipeline_run("c2")["acquired"] is True
