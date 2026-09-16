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

        rows = tg.topic_rows(tg.load_rows(gaps_db, "c1"), 0.70)
        order = [r["topic"] for r in tg.sort_rows(rows, "engagement")]

        # Must go through sort_rows: topic_rows no longer orders its output,
        # so asserting on it would pass on dict insertion order alone.
        assert order == ["Known", "Unknown"]
        assert {r["topic"]: r["median_engagement"] for r in rows}["Unknown"] is None


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


def _set_subtopics(db_path, pairs):
    conn = duckdb.connect(db_path)
    try:
        for unit_id, sub in pairs:
            conn.execute("UPDATE message_units SET subtopic = ? WHERE unit_id = ?",
                         [sub, unit_id])
    finally:
        conn.close()


class TestSubtopicGrouping:
    """The finer grain the extractor already records.

    At topic level, 15 accounts cover nearly every topic on the real corpus and
    engagement spans 3.2-5.0% — there is no gap visible because the taxonomy is
    too coarse. "Interview" is saturated; "the STAR method, explained properly"
    is the actual question, and it lives one level down.
    """

    def test_grouping_by_subtopic_splits_a_saturated_topic(self, gaps_db):
        tg = _tg()
        _seed(gaps_db, "c1", "a1", "p1", "Interview",
              ["Use the STAR method to structure answers"], 9.0)
        _seed(gaps_db, "c1", "a2", "p2", "Interview",
              ["Research the company before you go in"], 1.0)
        _set_subtopics(gaps_db, [("p1_u0", "star_method"), ("p2_u0", "research")])

        by_topic = tg.topic_rows(tg.load_rows(gaps_db, "c1", "topic"), 0.70)
        by_sub = tg.topic_rows(tg.load_rows(gaps_db, "c1", "subtopic"), 0.70)

        assert [r["topic"] for r in by_topic] == ["Interview"]
        assert sorted(r["topic"] for r in by_sub) == ["research", "star_method"]
        # The split is the point: one topic at 5% median hides a 9% and a 1%.
        assert by_topic[0]["median_engagement"] == 5.0
        assert {r["median_engagement"] for r in by_sub} == {9.0, 1.0}

    def test_a_null_subtopic_is_unclassified_not_dropped(self, gaps_db):
        """Absent is not absent-from-the-report.

        Silently dropping units with no subtopic would shrink the corpus
        without saying so, and the reader would never know the denominator
        moved.
        """
        tg = _tg()
        _seed(gaps_db, "c1", "a1", "p1", "Interview", ["A claim with no subtopic"], 5.0)

        rows = tg.topic_rows(tg.load_rows(gaps_db, "c1", "subtopic"), 0.70)

        assert [r["topic"] for r in rows] == ["(unclassified)"]
        assert rows[0]["units"] == 1

    def test_an_unknown_level_is_refused_rather_than_interpolated(self, gaps_db):
        tg = _tg()
        with pytest.raises(SystemExit) as exc:
            tg.load_rows(gaps_db, "c1", "claim")
        assert "topic or subtopic" in str(exc.value)


class TestImplausibleEngagementIsNotAMeasurement:
    """The real corpus contains values down to -0.64% and one at 135.71%.

    Engagement is likes as a percentage of views. Negative is impossible.
    Neither is a low performer nor a star, and sorting them into a ranking
    puts data errors at both ends of a report about market structure.
    """

    def test_a_negative_rate_is_excluded_not_ranked_last(self, gaps_db):
        tg = _tg()
        _seed(gaps_db, "c1", "a1", "p1", "Resume", ["A claim about resumes"], -0.64)
        _seed(gaps_db, "c1", "a2", "p2", "Resume", ["A different resume claim"], 4.0)

        row, = tg.topic_rows(tg.load_rows(gaps_db, "c1"), 0.70)

        assert row["median_engagement"] == 4.0
        assert row["engagement_known"] == 1
        assert row["engagement_implausible"] == 1
        assert row["posts"] == 2

    def test_a_rate_above_one_hundred_percent_is_excluded_too(self, gaps_db):
        tg = _tg()
        _seed(gaps_db, "c1", "a1", "p1", "Visa", ["A claim about visas"], 135.71)

        row, = tg.topic_rows(tg.load_rows(gaps_db, "c1"), 0.70)

        assert row["median_engagement"] is None, "an impossible rate was reported"
        assert row["engagement_implausible"] == 1

    def test_the_report_says_how_many_were_dropped(self, gaps_db, capsys):
        tg = _tg()
        _seed(gaps_db, "c1", "a1", "p1", "Resume", ["A claim about resumes"], -0.5)
        _seed(gaps_db, "c1", "a2", "p2", "Resume", ["A different resume claim"], 4.0)

        tg.cmd_report(gaps_db, "c1", 0.70)
        out = capsys.readouterr().out

        assert "1 post(s) excluded" in out
        assert "data errors" in out


class TestMinimumObservations:
    def test_a_single_post_row_can_be_hidden(self, gaps_db, capsys):
        """One post's engagement is an anecdote, not a rate.

        On the real corpus five different subtopics all read 135.71% because
        they came from one post, and they topped the ranking.
        """
        tg = _tg()
        _seed(gaps_db, "c1", "a1", "p1", "Fluke", ["A one off claim here"], 90.0)
        for n in range(3):
            _seed(gaps_db, "c1", f"a{n}", f"q{n}", "Real",
                  [f"Substantive claim number {n} about hiring"], 4.0)

        tg.cmd_report(gaps_db, "c1", 0.70, "topic", 3)
        out = capsys.readouterr().out

        assert "Real" in out
        assert "Fluke" not in out, "a single-post row survived --min-posts 3"
        assert "1 topic(s) hidden" in out

    def test_hiding_everything_exits_with_the_best_available_count(self, gaps_db):
        tg = _tg()
        _seed(gaps_db, "c1", "a1", "p1", "Thin", ["A claim"], 4.0)

        with pytest.raises(SystemExit) as exc:
            tg.cmd_report(gaps_db, "c1", 0.70, "topic", 50)

        assert "The most any has is 1" in str(exc.value)


def _embed(db_path, unit_id, vector):
    conn = duckdb.connect(db_path)
    try:
        conn.execute("UPDATE message_units SET embedding = ?::FLOAT[1536] "
                     "WHERE unit_id = ?", [vector, unit_id])
    finally:
        conn.close()


def _vec(*leading):
    return list(leading) + [0.0] * (1536 - len(leading))


class TestClusteringSubtopics:
    """subtopic is a free-text label, not a taxonomy.

    2,156 distinct values across 5,361 units on the real corpus — about 2.5
    units each, most of them one unit from one post. Topic gives 11 buckets;
    the grain a coverage question needs is between them.
    """

    def test_near_identical_subtopics_collapse_into_one_category(self, gaps_db):
        tg = _tg()
        _seed(gaps_db, "c1", "a1", "p1", "Salary", ["Ask for more money"], 4.0)
        _seed(gaps_db, "c1", "a2", "p2", "Salary", ["Negotiate your offer"], 4.0)
        _set_subtopics(gaps_db, [("p1_u0", "salary_negotiation"),
                                 ("p2_u0", "negotiating_pay")])
        _embed(gaps_db, "p1_u0", _vec(1.0, 0.0))
        _embed(gaps_db, "p2_u0", _vec(0.99, 0.14))

        mapping = tg.cluster_subtopics(tg.load_rows(gaps_db, "c1", "subtopic"), 0.62)

        assert len(set(mapping.values())) == 1, mapping

    def test_unrelated_subtopics_stay_separate(self, gaps_db):
        tg = _tg()
        _seed(gaps_db, "c1", "a1", "p1", "Salary", ["Ask for more money"], 4.0)
        _seed(gaps_db, "c1", "a2", "p2", "Visa", ["Sponsorship rules changed"], 4.0)
        _set_subtopics(gaps_db, [("p1_u0", "salary_negotiation"), ("p2_u0", "visa_rules")])
        _embed(gaps_db, "p1_u0", _vec(1.0, 0.0))
        _embed(gaps_db, "p2_u0", _vec(0.0, 1.0))

        mapping = tg.cluster_subtopics(tg.load_rows(gaps_db, "c1", "subtopic"), 0.62)

        assert len(set(mapping.values())) == 2, mapping

    def test_clusters_are_named_after_their_largest_member(self, gaps_db):
        """A category named after a one-post label is unreadable."""
        tg = _tg()
        for n in range(3):
            _seed(gaps_db, "c1", "a1", f"big{n}", "Salary",
                  [f"Negotiation point number {n}"], 4.0)
            _set_subtopics(gaps_db, [(f"big{n}_u0", "salary_negotiation")])
            _embed(gaps_db, f"big{n}_u0", _vec(1.0, 0.0))
        _seed(gaps_db, "c1", "a2", "small", "Salary", ["One more pay point"], 4.0)
        _set_subtopics(gaps_db, [("small_u0", "obscure_pay_label")])
        _embed(gaps_db, "small_u0", _vec(0.99, 0.14))

        mapping = tg.cluster_subtopics(tg.load_rows(gaps_db, "c1", "subtopic"), 0.62)

        assert mapping["obscure_pay_label"] == "salary_negotiation"

    def test_an_unembedded_subtopic_maps_to_itself_rather_than_being_guessed(self, gaps_db):
        """Absent is not 'close enough to something'.

        Forcing an unplaceable subtopic into a cluster would file unrelated
        material under a category name, and dropping it would shrink the
        corpus without saying so.
        """
        tg = _tg()
        # The embedded subtopic must be the BIGGEST, so that mapping the
        # unembedded one to the first seed would be a visible mistake. With
        # both at one post the ordering is alphabetical and the two outcomes
        # coincide — a mutation forcing it into the first cluster survived
        # that fixture.
        for n in range(3):
            _seed(gaps_db, "c1", "a1", f"big{n}", "Salary", [f"Pay point {n}"], 4.0)
            _set_subtopics(gaps_db, [(f"big{n}_u0", "salary_negotiation")])
            _embed(gaps_db, f"big{n}_u0", _vec(1.0, 0.0))
        _seed(gaps_db, "c1", "a2", "p2", "Salary", ["Another pay claim"], 4.0)
        _set_subtopics(gaps_db, [("p2_u0", "no_vector")])

        mapping = tg.cluster_subtopics(tg.load_rows(gaps_db, "c1", "subtopic"), 0.62)

        assert mapping["salary_negotiation"] == "salary_negotiation"
        assert mapping["no_vector"] == "no_vector"


class TestSortingDefaultsToCoverage:
    def test_coverage_sort_ranks_by_distinct_accounts_not_engagement(self):
        """Engagement rate is largely a property of the account, not the topic.

        Larger followings produce lower percentages mechanically, so a topic's
        figure is substantially "which accounts happened to cover it".
        """
        tg = _tg()
        rows = [
            {"topic": "Wide", "accounts": 9, "posts": 20, "median_engagement": 1.0},
            {"topic": "Narrow", "accounts": 2, "posts": 3, "median_engagement": 9.0},
        ]

        assert [r["topic"] for r in tg.sort_rows(rows, "coverage")] == ["Wide", "Narrow"]
        assert [r["topic"] for r in tg.sort_rows(rows, "engagement")] == ["Narrow", "Wide"]


class TestMeanVector:
    """Tested directly: the embedding column is fixed-width, so mixed widths
    cannot arise through the database today. They can arise the moment a
    second embedding provider is used — creative-director-ai#29 records a
    Voyage adapter at 512 dimensions against a FLOAT[1536] column, with the
    resize migration never written. Averaging across widths would produce a
    vector that is meaningless rather than wrong in any detectable way.
    """

    def test_vectors_of_different_widths_refuse_to_average(self):
        tg = _tg()
        assert tg._mean_vector([[1.0, 0.0], [1.0, 0.0, 0.0]]) is None

    def test_all_missing_returns_none_rather_than_a_zero_vector(self):
        tg = _tg()
        assert tg._mean_vector([None, None]) is None
        assert tg._mean_vector([]) is None

    def test_missing_vectors_are_skipped_not_counted_as_origins(self):
        tg = _tg()
        # Counting None as [0, 0] would halve the mean.
        assert tg._mean_vector([[2.0, 4.0], None]) == [2.0, 4.0]
