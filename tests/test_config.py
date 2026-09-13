def _set_required_keys(monkeypatch):
    monkeypatch.setenv("APIFY_TOKEN", "apify")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "deepgram")
    monkeypatch.setenv("OPENAI_API_KEY", "openai")


def test_optional_provider_secret_is_not_in_use_when_absent(monkeypatch):
    from core.config import secret_status

    _set_required_keys(monkeypatch)
    monkeypatch.delenv("ASSEMBLYAI_API_KEY", raising=False)

    rows = {row["name"]: row for row in secret_status()}

    assert rows["ASSEMBLYAI_API_KEY"]["required"] is False
    assert rows["ASSEMBLYAI_API_KEY"]["configured"] is False
    assert rows["ASSEMBLYAI_API_KEY"]["status"] == "not-in-use"
    assert all(row["status"] != "missing-required" for row in rows.values())


def test_optional_provider_secret_can_be_configured(monkeypatch):
    from core.config import secret_status

    _set_required_keys(monkeypatch)
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "assemblyai")

    rows = {row["name"]: row for row in secret_status()}

    assert rows["ASSEMBLYAI_API_KEY"]["required"] is False
    assert rows["ASSEMBLYAI_API_KEY"]["configured"] is True
    assert rows["ASSEMBLYAI_API_KEY"]["status"] == "configured"
