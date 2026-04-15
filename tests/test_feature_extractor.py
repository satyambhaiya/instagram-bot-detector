"""
Unit tests for the feature extractor.
No model or network calls — pure Python logic.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.schemas.instagram import PostData, RawInstagramProfile
from app.services.feature_extractor import FEATURE_COLS, extract_features

_UTC = timezone.utc


def _make_profile(**kwargs) -> RawInstagramProfile:
    defaults = dict(
        username="testuser",
        full_name="Test User",
        biography="A test bio",
        profile_pic_url="https://example.com/pic.jpg",
        followers_count=1000,
        following_count=500,
        posts_count=50,
        is_verified=False,
        is_private=False,
        is_business=False,
        external_url=None,
        recent_posts=[],
    )
    defaults.update(kwargs)
    return RawInstagramProfile(**defaults)


def _make_post(days_ago: int = 10, likes: int = 100, comments: int = 10,
               caption: str = "test", is_video: bool = False) -> PostData:
    return PostData(
        likes_count=likes,
        comments_count=comments,
        caption=caption,
        is_video=is_video,
        media_type="Video" if is_video else "Image",
        timestamp=datetime.now(_UTC) - timedelta(days=days_ago),
    )


class TestFeatureExtraction:
    def test_returns_all_24_features(self):
        profile = _make_profile()
        result = extract_features(profile)
        assert set(result.features.keys()) == set(FEATURE_COLS)

    def test_direct_features_are_correct(self):
        profile = _make_profile(followers_count=5000, following_count=200, posts_count=100)
        result = extract_features(profile)
        assert result.features["followers_count"] == 5000
        assert result.features["following_count"] == 200
        assert result.features["posts_count"] == 100

    def test_username_features(self):
        profile = _make_profile(username="user1234")
        result = extract_features(profile)
        assert result.features["username_length"] == len("user1234")
        # 4 digits out of 8 chars = 0.5
        assert result.features["username_digit_ratio"] == pytest.approx(0.5, abs=0.01)

    def test_follower_following_ratio(self):
        profile = _make_profile(followers_count=1000, following_count=500)
        result = extract_features(profile)
        assert result.features["follower_following_ratio"] == pytest.approx(2.0, abs=0.01)

    def test_bio_features(self):
        bio = "Travel photographer"
        profile = _make_profile(biography=bio, external_url="https://mysite.com")
        result = extract_features(profile)
        assert result.features["bio_length"] == len(bio)
        assert result.features["has_url_in_bio"] == 1.0

    def test_profile_pic_present(self):
        profile = _make_profile(profile_pic_url="https://example.com/pic.jpg")
        assert extract_features(profile).features["profile_pic"] == 1.0

    def test_profile_pic_absent(self):
        profile = _make_profile(profile_pic_url=None)
        assert extract_features(profile).features["profile_pic"] == 0.0

    def test_post_aggregations_with_posts(self):
        posts = [
            _make_post(likes=200, comments=20, caption="Hello #travel @friend"),
            _make_post(likes=100, comments=10, caption="Another post #food"),
            _make_post(likes=300, comments=30, caption="Video post", is_video=True),
        ]
        profile = _make_profile(recent_posts=posts)
        result = extract_features(profile)

        assert result.features["avg_likes_per_post"] == pytest.approx(200.0, abs=0.1)
        assert result.features["avg_comments_per_post"] == pytest.approx(20.0, abs=0.1)
        assert result.features["reels_ratio"] == pytest.approx(1 / 3, abs=0.01)

    def test_hashtag_and_mention_counting(self):
        posts = [
            _make_post(caption="#travel #food @alice"),
            _make_post(caption="#nature @bob @carol"),
        ]
        profile = _make_profile(recent_posts=posts)
        result = extract_features(profile)
        assert result.features["hashtags_per_post"] == pytest.approx(1.5, abs=0.01)
        assert result.features["mentions_per_post"] == pytest.approx(1.5, abs=0.01)

    def test_no_posts_adds_to_estimated(self):
        profile = _make_profile(recent_posts=[])
        result = extract_features(profile)
        assert "avg_likes_per_post" in result.estimated
        assert "night_activity_ratio" in result.estimated

    def test_always_has_estimated_features(self):
        # story_frequency, comment_reply_rate, unique_commenters_ratio are always estimated
        profile = _make_profile()
        result = extract_features(profile)
        for feat in ("story_frequency", "comment_reply_rate", "unique_commenters_ratio"):
            assert feat in result.estimated

    def test_all_features_are_finite_floats(self):
        posts = [_make_post() for _ in range(6)]
        profile = _make_profile(recent_posts=posts)
        result = extract_features(profile)
        for name, val in result.features.items():
            assert isinstance(val, float), f"{name} is not float"
            assert val == val, f"{name} is NaN"           # NaN check
            assert abs(val) < 1e9, f"{name} is unreasonably large"

    def test_zero_followers_does_not_crash(self):
        profile = _make_profile(followers_count=0, following_count=0)
        result = extract_features(profile)
        assert result.features["follower_following_ratio"] >= 0

    def test_private_account_is_flagged(self):
        profile = _make_profile(is_private=True, recent_posts=[])
        result = extract_features(profile)
        assert result.features["is_private"] == 1.0
