"""Tests for tools/topic_gaps.py — supply (corpus) joined to demand (engagement)."""
import importlib.util
import pathlib
import sys

import duckdb
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _tg():
    spec = importlib.util.spec_from_file_location("tg", ROOT / "tools" / "topic_gaps.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["tg"] = m
    spec.loader.exec_module(m)
    return m


@pytest.fixture
def gaps_db(tmp_path, monkeypatch):
    import core.db as db_mod

    monkeypatch.setattr(db_mod, "DB_PATH", str(tmp_path / "gaps.duckdb"))
    db_mod.init_db()
    return db_mod.DB_PATH


def _seed(db_path, client_id, account, post_id, topic, claims, engagement,
          username=None):
    conn = duckdb.connect(db_path)
    try:
        conn.execute(
            "INSERT OR IGNORE INTO creator_accounts VALUES "
            "(?,?,NULL,NULL,NULL,NULL,NULL,FALSE,NULL,NULL,NULL,now(),now())",
            [account, username or account])
        conn.execute(
            """INSERT INTO posts (post_id, client_id, account_id, engagement_rate,
               download_status, scraped_at, hashtags, mentions, video_url)
               VALUES (?,?,?,?, 'done', now(), [], [], 'u')""",
            [post_id, client_id, account, engagement])
        conn.execute(
            "INSERT INTO client_posts (client_id, post_id, added_at) VALUES (?,?,now())",
            [client_id, post_id])
        for i, claim in enumerate(claims):
            conn.execute(
                """INSERT INTO message_units (unit_id, client_id, post_id, text, claim,
                   topic, content_type, confidence, extracted_at, model)
                   VALUES (?,?,?,?,?,?, 'tip', 0.9, now(), 'm')""",
                [f"{post_id}_u{i}", client_id, post_id, claim, claim, topic])
    finally:
        conn.close()


class TestUnknownEngagementIsNotZero:
    """The column says DEFAULT 0, so this is the trap worth a test.

    engagement_rate_of() in core/db.py returns None when likes or views are
    missing, precisely so an unmeasured post does not read as an unpopular
    one. A topic assembled only from such posts must carry that through.
    """

    def test_a_topic_with_no_known_engagement_reports_no_data(self, gaps_db):
        tg = _tg()
        _seed(gaps_db, "c1", "a1", "p1", "Visa", ["Sponsorship rules changed"], None)

        row, = tg.topic_rows(tg.load_rows(gaps_db, "c1"), 0.70)

        assert row["median_engagement"] is None
        assert row["engagement_known"] == 0
        assert tg._pct(row["median_engagement"]) == tg.UNKNOWN

    def test_unknown_posts_do_not_drag_the_median_down(self, gaps_db):
        tg = _tg()
        _seed(gaps_db, "c1", "a1", "p1", "Resume", ["Quantify your bullets"], 8.0)
        _seed(gaps_db, "c1", "a2", "p2", "Resume", ["Lead with outcomes"], None)
        _seed(gaps_db, "c1", "a3", "p3", "Resume", ["Cut the summary"], 10.0)

        row, = tg.topic_rows(tg.load_rows(gaps_db, "c1"), 0.70)

        # Median of the two KNOWN values. Counting the unknown as 0 would give 8.0.
        assert row["median_engagement"] == 9.0
        assert row["engagement_known"] == 2
        assert row["posts"] == 3

    def test_unknown_topics_sort_last_rather_than_as_zero(self, gaps_db):
        tg = _tg()
        _seed(gaps_db, "c1", "a1", "p1", "Known", ["A claim about something"], 1.0)
        _seed(gaps_db, "c1", "a2", "p2", "Unknown", ["A different claim entirely"], None)

        order = [r["topic"] for r in tg.topic_rows(tg.load_rows(gaps_db, "c1"), 0.70)]

        # A 1.0% topic still outranks one with no data. Sorting unknown as 0
        # happens to give the same order here, so the value is asserted too.
        assert order == ["Known", "Unknown"]
        assert tg.topic_rows(tg.load_rows(gaps_db, "c1"), 0.70)[1]["median_engagement"] is None


class TestCountingIdeasNotUnits:
    _CV_TRIPLE = [
        "Colorful templates and selfies are not suitable for a professional CV",
        "A black and white template is more effective for a CV than a colorful one",
        "A standardized black and white template is easier to scan and professional",
    ]

    def test_stem_overlap_cannot_collapse_the_real_cv_triple(self):
        """Documents the fallback's ceiling rather than hiding it.

        These three are real units from one 10-second CV video, which a human
        reads as one idea stated three ways. Their pairwise stem overlaps are
        0.43, 0.29 and 0.43. The threshold that collapses them (0.40) flags
        81.5% of the real corpus as duplicated; the one that leaves the corpus
        intact (0.70) leaves all three standing. No cut-off separates "same
        idea" from "same topic" lexically, which is why embeddings exist.

        If this ever collapses to 1 at 0.70, the stemmer changed and the
        threshold calibration needs redoing.
        """
        tg = _tg()
        assert tg.distinct_ideas(self._CV_TRIPLE, 0.70) == (3, "stem overlap")
        # Even at 0.40 it only reaches 2, not 1. Greedy clustering compares
        # each claim against cluster REPRESENTATIVES, and the second claim
        # joins the first rather than becoming one, so the third — which
        # overlaps the second at 0.43 but the first at only 0.29 — never meets
        # it. Exact clustering would help slightly; it would not fix the
        # lexical ceiling that makes 0.40 necessary in the first place.
        assert tg.distinct_ideas(self._CV_TRIPLE, 0.40) == (2, "stem overlap")

    def test_embeddings_collapse_what_stem_overlap_cannot(self):
        """Near-identical vectors are one idea regardless of wording."""
        tg = _tg()
        base = [1.0, 0.0, 0.0]
        near = [0.99, 0.14, 0.0]
        far = [0.0, 0.0, 1.0]

        assert tg.distinct_ideas(["a", "b"], 0.90, [base, near]) == (1, "embedding")
        assert tg.distinct_ideas(["a", "b"], 0.90, [base, far]) == (2, "embedding")

    def test_a_single_missing_embedding_falls_back_rather_than_guessing(self):
        """Absent is not zero: a NULL vector is not the origin.

        Treating a missing embedding as a zero vector would give it cosine 0
        against everything, so it would read as a brand new idea every time
        and silently inflate the count.
        """
        tg = _tg()
        ideas, measure = tg.distinct_ideas(["a", "b"], 0.90, [[1.0, 0.0], None])
        assert measure == "stem overlap"

    def test_genuinely_different_claims_stay_separate(self, gaps_db):
        tg = _tg()
        _seed(gaps_db, "c1", "a1", "p1", "Mixed", [
            "Quantify achievements with concrete numbers",
            "Recruiters review Monday applications first",
            "Never disclose personal details during interviews",
        ], 5.0)

        row, = tg.topic_rows(tg.load_rows(gaps_db, "c1"), 0.70)

        assert row["ideas"] == 3
        assert row["measure"] == "stem overlap"


class TestCountingAccountsNotPosts:
    def test_one_creator_posting_repeatedly_is_one_voice(self, gaps_db):
        """Saturation is how many different people cover something."""
        tg = _tg()
        for n in range(4):
            _seed(gaps_db, "c1", "a1", f"p{n}", "Salary",
                  [f"Negotiation advice variant number {n} about pay"], 5.0)

        row, = tg.topic_rows(tg.load_rows(gaps_db, "c1"), 0.70)

        assert row["accounts"] == 1
        assert row["posts"] == 4


class TestEngagementWeighting:
    def test_a_post_weighs_once_however_many_units_it_produced(self, gaps_db):
        """Otherwise a verbose post decides its topic's engagement figure."""
        tg = _tg()
        _seed(gaps_db, "c1", "a1", "p1", "Interview",
              [f"Interview point number {i} covering distinct ground" for i in range(9)], 1.0)
        _seed(gaps_db, "c1", "a2", "p2", "Interview", ["A single different point"], 9.0)

        row, = tg.topic_rows(tg.load_rows(gaps_db, "c1"), 0.70)

        # Median of [1.0, 9.0]. Weighting by unit would give 1.0.
        assert row["median_engagement"] == 5.0


class TestClientScoping:
    def test_another_clients_posts_are_not_counted(self, gaps_db):
        tg = _tg()
        _seed(gaps_db, "c1", "a1", "p1", "Resume", ["Quantify your bullets"], 5.0)
        _seed(gaps_db, "c2", "a2", "p2", "Resume", ["Something else entirely here"], 50.0)

        row, = tg.topic_rows(tg.load_rows(gaps_db, "c1"), 0.70)

        assert row["accounts"] == 1
        assert row["posts"] == 1
        assert row["median_engagement"] == 5.0


def test_report_renders_no_data_not_zero(gaps_db, capsys):
    tg = _tg()
    _seed(gaps_db, "c1", "a1", "p1", "Visa", ["Sponsorship rules changed"], None)

    assert tg.cmd_report(gaps_db, "c1", 0.70) == 0
    out = capsys.readouterr().out

    assert "no data" in out
    assert "0.00%" not in out, "an unmeasured topic rendered as a real zero rate"


def test_empty_client_exits_rather_than_printing_an_empty_table(gaps_db):
    tg = _tg()
    with pytest.raises(SystemExit) as exc:
        tg.cmd_report(gaps_db, "nobody", 0.70)
    assert "nobody" in str(exc.value)


def test_report_warns_when_it_fell_back_to_stem_overlap(gaps_db, capsys):
    """A number produced by the weaker measure must say so.

    Reporting an idea count without naming the measure invites the reader to
    trust a figure that the CV-triple test shows is systematically too high.
    """
    tg = _tg()
    _seed(gaps_db, "c1", "a1", "p1", "Resume", ["Quantify your bullets"], 5.0)

    tg.cmd_report(gaps_db, "c1", 0.70)
    out = capsys.readouterr().out

    assert "stem overlap" in out
    assert "under-counts" in out, "the fallback's limitation was not stated"


def test_a_unit_whose_post_is_not_in_client_posts_is_invisible(gaps_db):
    """client_posts is the visibility list, and it is load-bearing on its own.

    message_units.client_id and client_posts are two separate filters. A unit
    can carry the right client_id while its post is listed under a DIFFERENT
    client — a shared competitor account, say.

    The discriminating case took two attempts. A post with no client_posts row
    at all is excluded by the join itself, so removing the cp.client_id
    predicate still passed. It has to be a post visible to another client.
    """
    tg = _tg()
    _seed(gaps_db, "c1", "a1", "p1", "Resume", ["Quantify your bullets"], 5.0)

    conn = duckdb.connect(gaps_db)
    try:
        conn.execute(
            """INSERT INTO posts (post_id, client_id, account_id, engagement_rate,
               download_status, scraped_at, hashtags, mentions, video_url)
               VALUES ('p_orphan','c1','a1', 99.0, 'done', now(), [], [], 'u')""")
        # Visible to c2, not to c1 — this is what the predicate guards.
        conn.execute(
            "INSERT INTO client_posts (client_id, post_id, added_at) "
            "VALUES ('c2','p_orphan', now())")
        conn.execute(
            """INSERT INTO message_units (unit_id, client_id, post_id, text, claim,
               topic, content_type, confidence, extracted_at, model)
               VALUES ('u_orphan','c1','p_orphan','x','Dropped account leftover claim',
                       'Resume','tip',0.9, now(),'m')""")
    finally:
        conn.close()

    row, = tg.topic_rows(tg.load_rows(gaps_db, "c1"), 0.70)

    assert row["posts"] == 1, "a post outside client_posts was counted"
    assert row["units"] == 1
    assert row["median_engagement"] == 5.0, "the orphan post's engagement leaked in"
