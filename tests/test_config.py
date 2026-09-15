import pytest

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


class TestSecretsTomlFallbackForCliCallers:
    """The CLI could not read the file the README tells you to use.

    `get_secret` only consulted `.streamlit/secrets.toml` when handed a
    Streamlit module. Every CLI tool calls it without one, so `reextract.py`
    refused to run on a machine where OPENAI_API_KEY was correctly configured
    in the documented place. Environment still wins; this is a fallback only.
    """

    def _cfg(self):
        import importlib, core.config
        return importlib.reload(core.config)

    def test_reads_the_key_from_secrets_toml(self, tmp_path, monkeypatch):
        cfg = self._cfg()
        toml = tmp_path / "secrets.toml"
        toml.write_text('OPENAI_API_KEY = "sk-from-file"\n', encoding="utf-8")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert cfg._secret_from_toml("OPENAI_API_KEY", toml) == "sk-from-file"

    def test_environment_wins_over_the_file(self, tmp_path, monkeypatch):
        cfg = self._cfg()
        toml = tmp_path / "secrets.toml"
        toml.write_text('OPENAI_API_KEY = "sk-from-file"\n', encoding="utf-8")
        monkeypatch.setattr(cfg, "_GLOBAL_SECRETS_TOML", tmp_path / "no-global.toml")
        monkeypatch.setattr(cfg, "_SECRETS_TOML", toml)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
        # An export is a deliberate act for one command; the file is ambient.
        assert cfg.get_secret("OPENAI_API_KEY") == "sk-from-env"

    def test_file_is_used_when_environment_is_empty(self, tmp_path, monkeypatch):
        cfg = self._cfg()
        toml = tmp_path / "secrets.toml"
        toml.write_text('OPENAI_API_KEY = "sk-from-file"\n', encoding="utf-8")
        monkeypatch.setattr(cfg, "_GLOBAL_SECRETS_TOML", tmp_path / "no-global.toml")
        monkeypatch.setattr(cfg, "_SECRETS_TOML", toml)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert cfg.get_secret("OPENAI_API_KEY") == "sk-from-file"

    @pytest.mark.parametrize("content", ["", "not valid toml ===", 'OTHER = "x"\n'])
    def test_unusable_file_means_not_configured_not_a_crash(
        self, tmp_path, monkeypatch, content
    ):
        cfg = self._cfg()
        toml = tmp_path / "secrets.toml"
        toml.write_text(content, encoding="utf-8")
        monkeypatch.setattr(cfg, "_GLOBAL_SECRETS_TOML", tmp_path / "no-global.toml")
        monkeypatch.setattr(cfg, "_SECRETS_TOML", toml)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        # Missing, malformed and absent-key all mean the same thing to a
        # caller. has_secret() must stay honest rather than raise.
        assert cfg.get_secret("OPENAI_API_KEY") == ""
        assert cfg.has_secret("OPENAI_API_KEY") is False

    def test_missing_file_means_not_configured(self, tmp_path, monkeypatch):
        cfg = self._cfg()
        monkeypatch.setattr(cfg, "_GLOBAL_SECRETS_TOML", tmp_path / "no-global.toml")
        monkeypatch.setattr(cfg, "_SECRETS_TOML", tmp_path / "nope.toml")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert cfg.get_secret("OPENAI_API_KEY") == ""


class TestBothStreamlitSecretsLocations:
    """Streamlit reads two files; the CLI must look in both.

    The first version of this fallback checked only the project-local file.
    A key living in ~/.streamlit/secrets.toml was therefore invisible to every
    CLI tool while the Streamlit app found it without trouble — the original
    bug again, one directory over.
    """

    def _cfg(self):
        import importlib, core.config
        return importlib.reload(core.config)

    def test_key_in_the_user_level_file_is_found(self, tmp_path, monkeypatch):
        cfg = self._cfg()
        glob = tmp_path / "global.toml"
        glob.write_text('OPENAI_API_KEY = "sk-from-home"\n', encoding="utf-8")
        monkeypatch.setattr(cfg, "_GLOBAL_SECRETS_TOML", glob)
        monkeypatch.setattr(cfg, "_SECRETS_TOML", tmp_path / "absent.toml")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert cfg.get_secret("OPENAI_API_KEY") == "sk-from-home"

    def test_project_file_wins_over_the_user_level_one(self, tmp_path, monkeypatch):
        cfg = self._cfg()
        glob = tmp_path / "global.toml"
        proj = tmp_path / "project.toml"
        glob.write_text('OPENAI_API_KEY = "sk-from-home"\n', encoding="utf-8")
        proj.write_text('OPENAI_API_KEY = "sk-from-project"\n', encoding="utf-8")
        monkeypatch.setattr(cfg, "_GLOBAL_SECRETS_TOML", glob)
        monkeypatch.setattr(cfg, "_SECRETS_TOML", proj)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        # The more specific file wins. This mirrors Streamlit's documented
        # behaviour; it is a claim about another project, not something
        # verified here, so check it if the two ever disagree in practice.
        assert cfg.get_secret("OPENAI_API_KEY") == "sk-from-project"

    def test_keys_split_across_the_two_files_are_both_found(self, tmp_path, monkeypatch):
        cfg = self._cfg()
        glob = tmp_path / "global.toml"
        proj = tmp_path / "project.toml"
        glob.write_text('OPENAI_API_KEY = "sk-from-home"\n', encoding="utf-8")
        proj.write_text('APIFY_TOKEN = "apify-from-project"\n', encoding="utf-8")
        monkeypatch.setattr(cfg, "_GLOBAL_SECRETS_TOML", glob)
        monkeypatch.setattr(cfg, "_SECRETS_TOML", proj)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("APIFY_TOKEN", raising=False)
        # This is the real shape of the machine that found the bug: some keys
        # in one file, some in the other.
        assert cfg.get_secret("OPENAI_API_KEY") == "sk-from-home"
        assert cfg.get_secret("APIFY_TOKEN") == "apify-from-project"

    def test_environment_still_beats_both_files(self, tmp_path, monkeypatch):
        cfg = self._cfg()
        glob = tmp_path / "global.toml"
        glob.write_text('OPENAI_API_KEY = "sk-from-home"\n', encoding="utf-8")
        monkeypatch.setattr(cfg, "_GLOBAL_SECRETS_TOML", glob)
        monkeypatch.setattr(cfg, "_SECRETS_TOML", tmp_path / "absent.toml")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
        assert cfg.get_secret("OPENAI_API_KEY") == "sk-from-env"
