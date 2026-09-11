"""Tests for the acquisition adapter (creative-director-ai #15)."""
from unittest.mock import MagicMock, patch

import pytest
import requests

from collection.sources.apify import ApifyError, ApifySource, to_record
from collection.sources.base import ReelRecord, ReelSource


class TestUnavailableVersusZero:
    """The rule the adapter exists to enforce."""

    def test_zero_is_not_missing(self):
        r = to_record({"shortCode": "a", "likesCount": 0, "videoViewCount": 0})
        assert r.likes == 0 and r.has("likes")
        assert r.views == 0 and r.has("views")
        assert r.unavailable() == ["comments_count", "duration_sec"]

    def test_missing_stays_none_not_zero(self):
        r = to_record({"shortCode": "a"})
        assert r.views is None, "a missing view count must not become 0"
        assert not r.has("views")
        assert set(r.unavailable()) == {"likes", "views", "comments_count", "duration_sec"}

    def test_explicit_null_stays_none(self):
        r = to_record({"shortCode": "a", "videoViewCount": None})
        assert r.views is None

    def test_unparseable_value_is_unavailable_not_zero(self):
        r = to_record({"shortCode": "a", "likesCount": "not a number"})
        assert r.likes is None, "garbage must read as unknown, never as zero"

    def test_full_coverage_reports_nothing_unavailable(self):
        r = to_record({
            "shortCode": "a", "likesCount": 5, "videoViewCount": 9,
            "commentsCount": 1, "videoDuration": 30.0,
        })
        assert r.unavailable() == []


class TestMapping:
    def test_maps_apify_field_names(self):
        r = to_record({
            "shortCode": "XYZ", "caption": "hello", "likesCount": 10,
            "videoViewCount": 100, "commentsCount": 3, "videoDuration": 29.5,
            "timestamp": "2026-01-01T00:00:00Z", "videoUrl": "http://v",
            "audioUrl": "http://a", "displayUrl": "http://t",
        }, handle="acct")
        assert (r.post_id, r.handle, r.source) == ("XYZ", "acct", "apify")
        assert (r.likes, r.views, r.comments_count, r.duration_sec) == (10, 100, 3, 29.5)
        assert (r.video_url, r.audio_url, r.thumbnail_url) == ("http://v", "http://a", "http://t")

    def test_handle_falls_back_to_owner_username(self):
        assert to_record({"shortCode": "a", "ownerUsername": "owner"}).handle == "owner"

    def test_raw_payload_is_retained(self):
        item = {"shortCode": "a", "somethingUnmapped": 1}
        assert to_record(item).raw is item


class TestApifySource:
    def test_satisfies_the_protocol(self):
        assert isinstance(ApifySource("token"), ReelSource)

    def test_empty_token_rejected(self):
        with pytest.raises(ApifyError):
            ApifySource("")

    def _resp(self, status=200, payload=None):
        m = MagicMock()
        m.status_code = status
        m.raise_for_status.return_value = None
        m.json.return_value = payload if payload is not None else []
        return m

    def test_fetch_returns_records(self):
        payload = [{"shortCode": "a", "likesCount": 1}, {"shortCode": "b"}]
        with patch("collection.sources.apify.requests.post", return_value=self._resp(payload=payload)):
            out = ApifySource("t").fetch_account_reels("acct", 10)
        assert [r.post_id for r in out] == ["a", "b"]
        assert all(isinstance(r, ReelRecord) for r in out)
        assert out[1].likes is None, "second item had no likes; must stay unknown"

    def test_auth_error_is_not_retried(self):
        resp = self._resp(status=401)
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError("401")
        with patch("collection.sources.apify.requests.post", return_value=resp) as m:
            with pytest.raises(ApifyError, match="Auth error"):
                ApifySource("t", retries=2).fetch_raw("acct")
        assert m.call_count == 1, "auth failures will not resolve on retry"

    def test_non_list_response_rejected(self):
        with patch("collection.sources.apify.requests.post", return_value=self._resp(payload={"x": 1})):
            with pytest.raises(ApifyError, match="Unexpected response format"):
                ApifySource("t", retries=0).fetch_raw("acct")
