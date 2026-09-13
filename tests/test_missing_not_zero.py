"""Missing counts must stay unknown, never become zero (creative-director-ai #21).

`or 0` made "the source did not supply this" and "this is genuinely zero"
indistinguishable. Engagement computed from a false zero is wrong in a way
nobody notices, because zero is a plausible number.
"""
import pytest


@pytest.fixture
def db(tmp_path, monkeypatch):
    import core.db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(tmp_path / "t.duckdb"))
    db_mod.init_db()
    return db_mod


@pytest.fixture
def account(db):
    return db.upsert_creator_account("acct", {"ownerUsername": "acct"})


def _get(db, post_id, cols="likes, views, comments_count, duration_sec, engagement_rate"):
    conn = db.get_connection()
    try:
        return conn.execute(f"SELECT {cols} FROM posts WHERE post_id = ?", [post_id]).fetchone()
    finally:
        conn.close()


class TestMissingStaysUnknown:
    def test_absent_view_count_is_null_not_zero(self, db, account):
        db.upsert_post(account, {"shortCode": "a", "likesCount": 10}, "c1")
        likes, views, *_ = _get(db, "a")
        assert likes == 10
        assert views is None, "an absent view count must not read as zero views"

    def test_genuine_zero_is_preserved(self, db, account):
        db.upsert_post(account, {"shortCode": "b", "likesCount": 0, "videoViewCount": 0}, "c1")
        likes, views, *_ = _get(db, "b")
        assert likes == 0 and views == 0, "a real zero is data, not absence"

    def test_all_counts_absent(self, db, account):
        db.upsert_post(account, {"shortCode": "c"}, "c1")
        likes, views, comments, duration, _ = _get(db, "c")
        assert (likes, views, comments, duration) == (None, None, None, None)

    def test_unparseable_reads_as_unknown(self, db, account):
        db.upsert_post(account, {"shortCode": "d", "likesCount": "many"}, "c1")
        assert _get(db, "d")[0] is None, "garbage must not be rounded down to zero"


class TestEngagementRate:
    def test_none_when_views_unknown(self, db, account):
        db.upsert_post(account, {"shortCode": "e", "likesCount": 50}, "c1")
        assert _get(db, "e")[4] is None, "unknown is not the same as no engagement"

    def test_none_when_likes_unknown(self, db, account):
        db.upsert_post(account, {"shortCode": "f", "videoViewCount": 1000}, "c1")
        assert _get(db, "f")[4] is None

    def test_computed_when_both_known(self, db, account):
        db.upsert_post(account, {"shortCode": "g", "likesCount": 50, "videoViewCount": 1000}, "c1")
        assert _get(db, "g")[4] == pytest.approx(5.0)

    def test_zero_views_does_not_divide(self, db, account):
        db.upsert_post(account, {"shortCode": "h", "likesCount": 5, "videoViewCount": 0}, "c1")
        assert _get(db, "h")[4] is None

    def test_genuine_zero_engagement_is_zero_not_none(self, db, account):
        db.upsert_post(account, {"shortCode": "i", "likesCount": 0, "videoViewCount": 500}, "c1")
        assert _get(db, "i")[4] == 0.0, "nobody engaged is a real, knowable result"


class TestUpdatePath:
    def test_rescrape_can_fill_in_a_previously_unknown_count(self, db, account):
        db.upsert_post(account, {"shortCode": "j", "likesCount": 1}, "c1")
        assert _get(db, "j")[1] is None
        db.upsert_post(account, {"shortCode": "j", "likesCount": 2, "videoViewCount": 900}, "c1")
        likes, views, *_ = _get(db, "j")
        assert (likes, views) == (2, 900)

    def test_rescrape_without_views_does_not_invent_zero(self, db, account):
        db.upsert_post(account, {"shortCode": "k", "likesCount": 1, "videoViewCount": 100}, "c1")
        db.upsert_post(account, {"shortCode": "k", "likesCount": 2}, "c1")
        assert _get(db, "k")[1] is None, "a later payload lacking views reports unknown, not zero"


class TestHelpersAreReusableOutsideIngestion:
    """The guards above only protect callers that go through `upsert_post`.

    Anything reading `raw_scrapes` JSON directly — the Data Viewer page does —
    re-derives these numbers itself and gets none of that protection. The
    helpers are public so those callers can reuse them instead of writing
    `or 0` again, which is how this bug arrived the first three times.
    """

    def test_helpers_are_public(self):
        from core.db import opt_number, engagement_rate_of
        assert callable(opt_number)
        assert callable(engagement_rate_of)

    def test_absent_stays_none(self):
        from core.db import opt_number
        assert opt_number(None, int) is None

    def test_genuine_zero_survives(self):
        from core.db import opt_number
        assert opt_number(0, int) == 0

    def test_unparseable_is_unknown_not_zero(self):
        from core.db import opt_number
        assert opt_number("not a number", int) is None
        assert opt_number([], float) is None

    def test_numeric_strings_still_parse(self):
        from core.db import opt_number
        assert opt_number("42", int) == 42
        assert opt_number("1.5", float) == 1.5

    @pytest.mark.parametrize(
        "likes, views",
        [(None, 100), (10, None), (None, None), (10, 0)],
    )
    def test_engagement_is_none_when_it_cannot_be_known(self, likes, views):
        from core.db import engagement_rate_of
        assert engagement_rate_of(likes, views) is None

    def test_zero_likes_with_known_views_is_a_real_zero(self):
        from core.db import engagement_rate_of
        # This is the case the `or 0` bug destroys in the other direction:
        # nobody engaged is a fact, and it must not read as unknown.
        assert engagement_rate_of(0, 100) == 0.0

    def test_engagement_is_computed_when_both_known(self):
        from core.db import engagement_rate_of
        assert engagement_rate_of(25, 100) == 25.0
