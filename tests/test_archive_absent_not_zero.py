import json
from pathlib import Path

import duckdb


def test_archived_predictor_keeps_missing_duration_unknown():
    from archive.engagement_predictor import EngagementPredictor

    predictor = object.__new__(EngagementPredictor)
    predictor.get_owner_avg_engagement = lambda _owner: None

    features = predictor.extract_features({"caption": "plain caption"})

    assert features["duration"] is None


def test_archived_predictor_skips_training_rows_with_unknown_engagement(tmp_path):
    from archive.engagement_predictor import EngagementPredictor

    db_path = tmp_path / "raw.duckdb"
    conn = duckdb.connect(str(db_path))
    conn.execute("CREATE TABLE raw_scrapes (client_id TEXT, raw TEXT)")
    conn.execute(
        "INSERT INTO raw_scrapes VALUES (?, ?)",
        ["client_a", json.dumps({"shortCode": "missing_likes", "videoViewCount": 100})],
    )
    conn.execute(
        "INSERT INTO raw_scrapes VALUES (?, ?)",
        ["client_a", json.dumps({"shortCode": "known_zero", "likesCount": 0, "videoViewCount": 100})],
    )
    conn.execute(
        "INSERT INTO raw_scrapes VALUES (?, ?)",
        ["client_a", json.dumps({"shortCode": "known_rate", "likesCount": 25, "videoViewCount": 100})],
    )

    predictor = object.__new__(EngagementPredictor)
    predictor.conn = conn
    predictor.client_id = "client_a"
    predictor.feature_names = []
    predictor.get_owner_avg_engagement = lambda _owner: None

    _features, labels = predictor.prepare_training_data()

    assert labels == [0.0, 25.0]
    conn.close()


def test_archived_predictor_owner_average_ignores_unknown_likes(tmp_path):
    from archive.engagement_predictor import EngagementPredictor

    db_path = tmp_path / "owner.duckdb"
    conn = duckdb.connect(str(db_path))
    conn.execute("CREATE TABLE raw_scrapes (client_id TEXT, raw TEXT)")
    conn.execute(
        "INSERT INTO raw_scrapes VALUES (?, ?)",
        [
            "client_a",
            json.dumps({"ownerUsername": "acct", "videoViewCount": 100}),
        ],
    )
    conn.execute(
        "INSERT INTO raw_scrapes VALUES (?, ?)",
        [
            "client_a",
            json.dumps({"ownerUsername": "acct", "likesCount": 10, "videoViewCount": 100}),
        ],
    )

    predictor = object.__new__(EngagementPredictor)
    predictor.conn = conn
    predictor.client_id = "client_a"

    assert predictor.get_owner_avg_engagement("acct") == 10.0
    conn.close()


def test_archived_predictor_owner_average_has_no_magic_default(tmp_path):
    from archive.engagement_predictor import EngagementPredictor

    db_path = tmp_path / "owner_unknown.duckdb"
    conn = duckdb.connect(str(db_path))
    conn.execute("CREATE TABLE raw_scrapes (client_id TEXT, raw TEXT)")
    conn.execute(
        "INSERT INTO raw_scrapes VALUES (?, ?)",
        [
            "client_a",
            json.dumps({"ownerUsername": "acct", "videoViewCount": 100}),
        ],
    )

    predictor = object.__new__(EngagementPredictor)
    predictor.conn = conn
    predictor.client_id = "client_a"

    assert predictor.get_owner_avg_engagement("acct") is None
    conn.close()


def test_archived_graph_raw_metrics_keep_missing_values_unknown():
    from archive.graph_analyzer import post_metrics_from_raw

    metrics = post_metrics_from_raw({"videoViewCount": 100})

    assert metrics["likes"] is None
    assert metrics["views"] == 100
    assert metrics["duration"] is None
    assert metrics["engagement_rate"] is None


def test_archived_graph_similarity_does_not_award_unknown_zero_matches():
    from archive.graph_analyzer import InstagramGraphAnalyzer

    analyzer = object.__new__(InstagramGraphAnalyzer)
    post1 = {
        "owner": "one",
        "hashtags": [],
        "mentions": [],
        "caption": "",
        "engagement_rate": None,
        "duration": None,
    }
    post2 = {
        "owner": "two",
        "hashtags": [],
        "mentions": [],
        "caption": "",
        "engagement_rate": None,
        "duration": None,
    }

    assert analyzer.calculate_content_similarity(post1, post2) == 0


def test_archived_video_analysis_result_keeps_missing_metrics_unknown():
    from archive.video_analysis_metrics import build_video_analysis_result

    row = build_video_analysis_result({"caption": "No metrics here"}, "abc", file_size_mb=None)

    assert row["file_size_mb"] is None
    assert row["likes"] is None
    assert row["views"] is None
    assert row["plays"] is None
    assert row["comments"] is None
    assert row["duration"] is None
    assert row["engagement_rate"] is None
    assert row["comments_rate"] is None
    assert row["play_completion"] is None


def test_archived_video_file_size_error_is_unknown_not_zero(tmp_path):
    from archive.video_analysis_metrics import video_file_size_mb

    assert video_file_size_mb(str(tmp_path / "missing.mp4")) is None


def test_archived_video_analysis_result_preserves_real_zero_engagement():
    from archive.video_analysis_metrics import build_video_analysis_result

    row = build_video_analysis_result(
        {"likesCount": 0, "videoViewCount": 100, "commentsCount": 0, "videoPlayCount": 0},
        "abc",
        file_size_mb=0,
    )

    assert row["file_size_mb"] == 0
    assert row["likes"] == 0
    assert row["views"] == 100
    assert row["engagement_rate"] == 0.0
    assert row["comments_rate"] is None
    assert row["play_completion"] == 0.0


def test_archived_video_analysis_formatters_say_unknown():
    from archive.video_analysis_metrics import format_count_or_unknown, format_percent_or_unknown

    assert format_count_or_unknown(None) == "unknown"
    assert format_percent_or_unknown(None) == "unknown"


def test_archived_video_page_no_longer_contains_metric_default_patterns():
    page = Path("archive/_3_🤖_AI_Video_Analysis.py").read_text(encoding="utf-8")

    forbidden = [
        "metadata.get('likesCount', 0)",
        "metadata.get('videoViewCount', 0)",
        "metadata.get('videoPlayCount', 0)",
        "metadata.get('commentsCount', 0)",
        "metadata.get('videoDuration', 0)",
        "file_size = 0",
        "else 0",
    ]
    for pattern in forbidden:
        assert pattern not in page
