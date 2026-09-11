"""Per-persona, versioned topics (creative-director-ai #25)."""
import pytest

from core.taxonomy import (
    CATCH_ALL_TOPIC_ID,
    create_taxonomy_version,
    get_active_taxonomy,
    list_topics,
    slugify_topic,
    topic_names,
    topics_for_prompt,
    validate_topic,
)


@pytest.fixture
def db(tmp_path, monkeypatch):
    import core.db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(tmp_path / "t.duckdb"))
    db_mod.init_db()
    return db_mod


@pytest.fixture
def personas(db):
    import core.clients as cl
    return cl.create_client("Ari AI", "AI"), cl.create_client("Ari Jobs", "careers")


AI_TOPICS = [
    {"name": "Agents", "description": "Agentic workflows, tool use"},
    {"name": "Tooling", "description": "Frameworks and developer tools"},
    {"name": "Adoption", "description": "How teams actually adopt AI"},
]


class TestPersonaIsolation:
    def test_each_persona_has_its_own_topics(self, personas):
        ai, jobs = personas
        create_taxonomy_version(ai, AI_TOPICS)
        create_taxonomy_version(jobs, [{"name": "Resume"}, {"name": "Interview"}])
        assert "Agents" in topic_names(ai)
        assert "Agents" not in topic_names(jobs)
        assert "Resume" in topic_names(jobs)
        assert "Resume" not in topic_names(ai)

    def test_demo_client_keeps_its_original_eleven(self, db):
        names = topic_names(db.DEFAULT_CLIENT_ID)
        assert len(names) == 11
        assert {"ATS", "Resume", "Visa", "General"} <= set(names)

    def test_a_new_persona_starts_empty(self, personas):
        """A new persona must not inherit somebody else's vocabulary."""
        ai, _ = personas
        assert topic_names(ai) == []

    def test_prompt_uses_the_persona_vocabulary(self, personas):
        ai, _ = personas
        create_taxonomy_version(ai, AI_TOPICS)
        prompt = topics_for_prompt(ai)
        assert "Agents" in prompt and "Agentic workflows" in prompt
        assert "Resume" not in prompt


class TestVersioning:
    def test_new_version_supersedes_without_deleting(self, personas):
        ai, _ = personas
        v1 = create_taxonomy_version(ai, AI_TOPICS)
        v2 = create_taxonomy_version(ai, AI_TOPICS + [{"name": "Evaluation"}])

        assert get_active_taxonomy(ai)["taxonomy_id"] == v2
        assert get_active_taxonomy(ai)["version"] == 2
        # v1 still readable — this is the whole point
        assert len(list_topics(ai, v1)) == 4   # 3 + catch-all
        assert len(list_topics(ai, v2)) == 5   # 4 + catch-all

    def test_rename_keeps_topic_id_so_lineage_survives(self, personas):
        ai, _ = personas
        v1 = create_taxonomy_version(ai, [{"name": "Agents"}])
        original = [t for t in list_topics(ai, v1) if t["name"] == "Agents"][0]

        v2 = create_taxonomy_version(
            ai, [{"topic_id": original["topic_id"], "name": "Agentic Workflows"}]
        )
        renamed = [t for t in list_topics(ai, v2) if t["topic_id"] == original["topic_id"]]
        assert renamed and renamed[0]["name"] == "Agentic Workflows"

    def test_every_version_gets_a_catch_all(self, personas):
        ai, _ = personas
        create_taxonomy_version(ai, AI_TOPICS)
        assert any(t["topic_id"] == CATCH_ALL_TOPIC_ID for t in list_topics(ai))

    def test_empty_taxonomy_rejected(self, personas):
        ai, _ = personas
        with pytest.raises(ValueError):
            create_taxonomy_version(ai, [])

    def test_duplicate_topic_ids_collapse(self, personas):
        ai, _ = personas
        create_taxonomy_version(ai, [{"name": "Agents"}, {"name": "agents"}])
        ids = [t["topic_id"] for t in list_topics(ai)]
        assert len(ids) == len(set(ids))


class TestValidateTopic:
    def test_resolves_against_the_right_persona(self, personas):
        ai, jobs = personas
        create_taxonomy_version(ai, AI_TOPICS)
        create_taxonomy_version(jobs, [{"name": "Resume"}])
        assert validate_topic(ai, "Agents")[1] == "Agents"
        # "Agents" is meaningless to the jobs persona -> its catch-all
        assert validate_topic(jobs, "Agents")[0] == CATCH_ALL_TOPIC_ID

    def test_slug_is_stable_and_readable(self):
        assert slugify_topic("Model Releases") == "model_releases"
        assert slugify_topic("  AI/ML  ") == "ai_ml"
        assert slugify_topic("") .startswith("topic_")
