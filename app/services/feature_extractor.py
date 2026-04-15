"""
Feature Extractor
=================
Converts a `RawInstagramProfile` (from any Instagram provider) into the 24
numerical features expected by the BotDetectorNet model.

Features that can be directly observed from the public API are computed from
real data. Features that require private data (story views, comment thread
details) are statistically estimated from the available signals.

Returns
-------
features : dict[str, float]
    All 24 feature values (unscaled, in their original domain).
estimated : list[str]
    Names of features that were estimated rather than directly observed.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from statistics import mean
from typing import NamedTuple

from app.schemas.social import PostData, RawSocialProfile

# Backward compat
RawInstagramProfile = RawSocialProfile

_UTC = timezone.utc

# Features the model was trained on (order matters for the vector)
FEATURE_COLS: list[str] = [
    "followers_count", "following_count", "posts_count",
    "avg_likes_per_post", "avg_comments_per_post",
    "posts_per_day", "follower_following_ratio", "engagement_rate",
    "profile_pic", "bio_length", "has_url_in_bio", "is_verified",
    "is_private", "account_age_days", "username_digit_ratio",
    "username_length", "night_activity_ratio", "avg_caption_length",
    "hashtags_per_post", "mentions_per_post", "story_frequency",
    "reels_ratio", "comment_reply_rate", "unique_commenters_ratio",
    # v2.1 — engagement quality + behavioral consistency + mass-follow detection
    "likes_comments_ratio",          # avg_likes / avg_comments (high = bought likes)
    "posting_regularity",            # 0 = fixed intervals (bot), 1 = irregular (human)
    "following_to_followers_ratio",  # following / followers (high = mass-follow bot)
]

# Night window: 22:00 – 06:00 UTC
_NIGHT_START = 22
_NIGHT_END = 6


class ExtractionResult(NamedTuple):
    features: dict[str, float]
    estimated: list[str]


# ── Public entry point ─────────────────────────────────────────────────────────

def extract_features(profile: RawInstagramProfile) -> ExtractionResult:
    """
    Main function called by the predictor service.
    Returns (features_dict, list_of_estimated_feature_names).
    """
    estimated: list[str] = []
    posts = profile.recent_posts

    # ── Direct profile features ───────────────────────────────────────────────
    followers = max(profile.followers_count, 0)
    following = max(profile.following_count, 0)
    posts_count = max(profile.posts_count, 0)
    has_profile_pic = 1 if profile.profile_pic_url else 0
    bio_length = len(profile.biography or "")
    has_url_in_bio = 1 if profile.external_url else 0
    is_verified = int(profile.is_verified)
    is_private = int(profile.is_private)

    # ── Username features ─────────────────────────────────────────────────────
    username = profile.username or ""
    username_length = max(len(username), 1)
    digit_count = sum(c.isdigit() for c in username)
    username_digit_ratio = round(digit_count / username_length, 4)

    # ── Account age ───────────────────────────────────────────────────────────
    account_age_days, age_estimated = _estimate_account_age(
        posts, posts_count, profile.account_created_at
    )
    if age_estimated:
        estimated.append("account_age_days")

    # ── Post-level aggregations ───────────────────────────────────────────────
    if posts:
        avg_likes = mean(p.likes_count for p in posts)
        avg_comments = mean(p.comments_count for p in posts)
        avg_caption_length = mean(len(p.caption) for p in posts)
        hashtags_per_post = mean(_count_hashtags(p.caption) for p in posts)
        mentions_per_post = mean(_count_mentions(p.caption) for p in posts)
        reels_ratio = sum(1 for p in posts if p.is_video) / len(posts)
        night_activity_ratio = _compute_night_ratio(posts)
        posting_regularity = _compute_posting_regularity(posts)
    elif posts_count > 0:
        # Account has posts but we couldn't fetch them (e.g. Twitter GraphQL
        # only returns profile data, not tweets). Instead of zeros (which
        # the model interprets as "no engagement" = bot), estimate
        # engagement proportional to followers so the model focuses on
        # the profile features it CAN see (followers, following, ratio,
        # username, bio, account age).
        avg_likes, avg_comments = _estimate_engagement_from_profile(followers)
        avg_caption_length = 80.0   # median post/tweet length
        hashtags_per_post = 1.0
        mentions_per_post = 0.9
        reels_ratio = 0.12
        night_activity_ratio = 0.14  # human median
        posting_regularity = 0.6    # neutral (no timestamps to compute from)
        estimated += [
            "avg_likes_per_post", "avg_comments_per_post", "avg_caption_length",
            "hashtags_per_post", "mentions_per_post", "reels_ratio", "night_activity_ratio",
            "posting_regularity",
        ]
    else:
        # Genuinely zero posts (brand-new or empty account)
        avg_likes = 0.0
        avg_comments = 0.0
        avg_caption_length = 0.0
        hashtags_per_post = 0.0
        mentions_per_post = 0.0
        reels_ratio = 0.0
        night_activity_ratio = 0.3   # neutral estimate
        posting_regularity = 0.5    # neutral
        estimated += [
            "avg_likes_per_post", "avg_comments_per_post", "avg_caption_length",
            "hashtags_per_post", "mentions_per_post", "reels_ratio", "night_activity_ratio",
            "posting_regularity",
        ]

    # ── Computed features ─────────────────────────────────────────────────────
    follower_following_ratio = min(followers / max(following, 1), 500.0)
    raw_er = avg_likes / max(followers, 1)  # match training formula (likes only)
    engagement_rate = _normalize_engagement_rate(raw_er, followers)
    posts_per_day = posts_count / max(account_age_days, 1)

    # v2.1 — Engagement quality + mass-follow detection
    likes_comments_ratio = min(avg_likes / max(avg_comments, 0.1), 100.0)
    following_to_followers_ratio = min(following / max(followers, 1), 100.0)

    # ── Detect automated behavior from observable signals ─────────────────────
    # High ppd alone isn't enough — large orgs (NASA, news sites) post often
    # with human teams.  Require a COMBINATION of automated patterns.
    _auto_signals = 0
    if posts_per_day > 10 and following == 0:
        _auto_signals += 2                     # broadcast-only + extreme volume
    if posts_per_day > 5 and night_activity_ratio > 0.40:
        _auto_signals += 2                     # high volume + round-the-clock
    if following == 0 and followers > 100 and posts_per_day > 2:
        _auto_signals += 1                     # broadcast pattern
    if night_activity_ratio > 0.50 and posts_per_day > 1:
        _auto_signals += 1                     # strong nocturnal posting
    if hashtags_per_post == 0 and mentions_per_post == 0 and posts_per_day > 5:
        _auto_signals += 1                     # zero diversity + high volume
    _is_automated = _auto_signals >= 2

    # ── Estimated features (not accessible via public API) ────────────────────
    story_frequency = _estimate_story_frequency(posts_per_day, engagement_rate, is_private)
    if _is_automated:
        # Automated accounts don't reply to comments or attract diverse commenters
        comment_reply_rate = 0.02
        unique_commenters_ratio = 0.15
    else:
        comment_reply_rate = _estimate_comment_reply_rate(engagement_rate, avg_comments, followers)
        unique_commenters_ratio = _estimate_unique_commenters_ratio(avg_comments, followers, engagement_rate)
    estimated += ["story_frequency", "comment_reply_rate", "unique_commenters_ratio"]

    features = {
        "followers_count":          float(followers),
        "following_count":          float(following),
        "posts_count":              float(posts_count),
        "avg_likes_per_post":       round(float(avg_likes), 4),
        "avg_comments_per_post":    round(float(avg_comments), 4),
        "posts_per_day":            round(posts_per_day, 4),
        "follower_following_ratio": round(follower_following_ratio, 4),
        "engagement_rate":          round(engagement_rate, 6),
        "profile_pic":              float(has_profile_pic),
        "bio_length":               float(bio_length),
        "has_url_in_bio":           float(has_url_in_bio),
        "is_verified":              float(is_verified),
        "is_private":               float(is_private),
        "account_age_days":         float(account_age_days),
        "username_digit_ratio":     round(username_digit_ratio, 4),
        "username_length":          float(username_length),
        "night_activity_ratio":     round(night_activity_ratio, 4),
        "avg_caption_length":       round(float(avg_caption_length), 2),
        "hashtags_per_post":        round(float(hashtags_per_post), 2),
        "mentions_per_post":        round(float(mentions_per_post), 2),
        "story_frequency":          round(story_frequency, 4),
        "reels_ratio":              round(reels_ratio, 4),
        "comment_reply_rate":       round(comment_reply_rate, 4),
        "unique_commenters_ratio":  round(unique_commenters_ratio, 4),
        # v2.1
        "likes_comments_ratio":     round(likes_comments_ratio, 4),
        "posting_regularity":       round(posting_regularity, 4),
        "following_to_followers_ratio": round(following_to_followers_ratio, 4),
    }

    return ExtractionResult(features=features, estimated=list(dict.fromkeys(estimated)))


# ── Internal helpers ───────────────────────────────────────────────────────────

def _normalize_engagement_rate(raw_er: float, followers: int) -> float:
    """Normalize engagement rate for large accounts.

    The model was trained on data where ER is in [0, 1] with median ≈ 1.0.
    In the real world, ER drops dramatically with follower count:
      <10K: 3-8%    10K-100K: 1-3%    100K-1M: 0.5-1.5%
      1M-10M: 0.2-0.5%    >10M: 0.05-0.2%

    For accounts under 10K, the raw ER is in a range the model handles.
    For larger accounts we progressively blend in a log-compressed
    engagement rate that maps very-low real-world ER into the training
    distribution.  The blend is smooth: 0% at 10K, 100% at 500K+.
    """
    import math

    if raw_er <= 0:
        return 0.05

    raw_capped = min(raw_er, 1.0)

    # Small accounts: raw ER is directly usable
    if followers <= 10_000:
        return raw_capped

    # Log-scale compression: maps [0.00001, 0.1] → [0.3, 1.0]
    log_er = math.log10(max(raw_er, 1e-7))
    log_er = max(min(log_er, -1), -5)
    normalized = 0.3 + (log_er + 5) / 4 * 0.7

    # Smooth blend: 0 at 10K followers → 1 at 500K+
    blend = min(max((math.log10(followers) - 4) / 1.7, 0.0), 1.0)
    return round(raw_capped * (1 - blend) + normalized * blend, 6)


def _estimate_engagement_from_profile(followers: int) -> tuple[float, float]:
    """Estimate avg_likes and avg_comments from follower count alone.

    When the scraper returns profile metadata but no posts (e.g. Twitter
    GraphQL), we need plausible engagement values so the model doesn't
    see all-zeros (which it learned = bot).  The target is an
    engagement_rate around 0.7–1.0 (human median in training data).

    The formula avg_likes ≈ followers × 0.7, avg_comments ≈ followers × 0.25
    gives ER = (0.7 + 0.25) = 0.95 regardless of follower count — right in
    the middle of the human distribution.  For very small accounts we set a
    floor so the values don't collapse to near-zero.
    """
    avg_likes = max(followers * 0.70, 5.0)
    avg_comments = max(followers * 0.25, 1.0)
    return avg_likes, avg_comments

def _estimate_account_age(
    posts: list[PostData],
    posts_count: int,
    account_created_at: datetime | None,
) -> tuple[int, bool]:
    """Return (age_in_days, was_estimated)."""
    now = datetime.now(_UTC)

    if account_created_at:
        age = max((now - account_created_at).days, 1)
        return age, False

    if posts:
        timestamped = [p for p in posts if p.timestamp is not None]
        if timestamped:
            oldest_post = min(p.timestamp for p in timestamped)  # type: ignore[arg-type]
            days_from_oldest = max((now - oldest_post).days, 1)
            # The scraped posts are typically the most recent ~12.
            # Extrapolate: if we have 12 posts spanning X days, the full
            # account is roughly posts_count/12 times older.
            n_scraped = len(timestamped)
            multiplier = posts_count / max(n_scraped, 1) if posts_count > n_scraped else 1.0
            estimated_age = max(int(days_from_oldest * multiplier), days_from_oldest)
            return estimated_age, True

    # Fallback: derive from posting frequency heuristic
    if posts_count > 0:
        # Assume average posting rate of ~0.5 posts/day for unknown accounts
        return max(int(posts_count / 0.5), 30), True

    return 365, True   # default: 1 year for accounts with no posts


def _count_hashtags(caption: str) -> int:
    return len(re.findall(r"#\w+", caption))


def _count_mentions(caption: str) -> int:
    return len(re.findall(r"@\w+", caption))


def _compute_night_ratio(posts: list[PostData]) -> float:
    """Fraction of posts published between 22:00 and 06:00 UTC."""
    timestamped = [p for p in posts if p.timestamp is not None]
    if not timestamped:
        return 0.3   # neutral estimate

    night_count = sum(
        1 for p in timestamped
        if p.timestamp.hour >= _NIGHT_START or p.timestamp.hour < _NIGHT_END  # type: ignore[union-attr]
    )
    return round(night_count / len(timestamped), 4)


def _compute_posting_regularity(posts: list[PostData]) -> float:
    """Compute posting regularity from post timestamps.

    Returns 0.0 (perfectly regular / bot-like) to 1.0 (irregular / human-like).
    Uses the coefficient of variation (std/mean) of time intervals between posts.

    Bots post at fixed, mechanical intervals → low CV → low regularity.
    Humans post at random times → high CV → high regularity.
    """
    timestamped = [p for p in posts if p.timestamp is not None]
    if len(timestamped) < 3:
        return 0.6  # not enough data → neutral

    sorted_ts = sorted(p.timestamp for p in timestamped)
    intervals = [
        (sorted_ts[i + 1] - sorted_ts[i]).total_seconds()
        for i in range(len(sorted_ts) - 1)
    ]

    intervals = [iv for iv in intervals if iv > 0]  # drop zero-length gaps
    if len(intervals) < 2:
        return 0.6

    mean_iv = sum(intervals) / len(intervals)
    if mean_iv < 1:
        return 0.05  # near-zero intervals → mechanical bot

    variance = sum((x - mean_iv) ** 2 for x in intervals) / len(intervals)
    cv = (variance ** 0.5) / mean_iv  # coefficient of variation

    # Normalize: CV 0 → 0.0 (regular), CV 2+ → 1.0 (irregular)
    return round(min(cv / 2.0, 1.0), 4)


def _estimate_story_frequency(
    posts_per_day: float, engagement_rate: float, is_private: int
) -> float:
    """
    Real-world calibrated story frequency estimate.

    Research shows average Instagram users post 1-2 stories/day.
    Bots post grid content excessively but almost never use Stories.
    The old formula (posts_per_day * 1.5) gave near-zero values for
    accounts posting every few days, incorrectly flagging them as bots.
    """
    # Clear bot signal: posting grid content at machine-like speed
    if posts_per_day > 5.0:
        return 0.08

    # Most active human accounts post stories regardless of grid post frequency.
    # Use engagement rate as the primary signal (better proxy than post cadence).
    if engagement_rate > 0.05:      # Excellent engagement — very active creator
        base = 2.0
    elif engagement_rate > 0.02:    # Good engagement — typical creator
        base = 1.5
    elif engagement_rate > 0.005:   # Average engagement — casual user
        base = 1.0
    else:
        base = 0.6                  # Low engagement — uncertain, use safe default

    if is_private:
        base = min(base + 0.4, 4.0)

    return round(base, 3)


def _estimate_comment_reply_rate(
    engagement_rate: float, avg_comments: float, followers: int
) -> float:
    """
    Real-world calibrated comment reply rate.

    Studies show smaller accounts reply more often (they have time and community).
    The old formula (ER * 8) gave ~0.10 for typical 1-3% ER accounts,
    far below the human training distribution (mean ~0.5), causing misclassification.

    Calibration (nano/micro/macro accounts):
      < 5 K followers  : reply ~40-65% of comments
      5 K–50 K         : reply ~25-45%
      50 K–500 K       : reply ~12-25%
      > 500 K          : reply ~5-15%
    """
    # Hard bot signal: near-zero engagement AND almost no comments
    if avg_comments < 0.5 and engagement_rate < 0.003:
        return 0.05

    # Account size tiers
    if followers < 5_000:
        base = 0.55 if engagement_rate >= 0.01 else 0.32
    elif followers < 50_000:
        base = 0.40 if engagement_rate >= 0.01 else 0.20
    elif followers < 500_000:
        base = 0.22 if engagement_rate >= 0.005 else 0.10
    else:
        base = 0.12 if engagement_rate >= 0.003 else 0.06

    return round(base, 4)


def _estimate_unique_commenters_ratio(
    avg_comments: float, followers: int, engagement_rate: float
) -> float:
    """
    Real-world calibrated unique commenter ratio.

    Real accounts attract diverse commenters. Bot comment pods recycle
    the same 5-20 fake accounts, giving a very low ratio.
    The old formula (avg_comments / followers*0.01) gave ~0.13 for typical
    accounts, which landed squarely in the bot range in training data.

    Calibration:
      High ER (>5%)  : ~0.80 unique  (authentic community)
      Good ER (2-5%) : ~0.68 unique
      Average ER (1-2%): ~0.55
      Low ER (<1%)   : ~0.40
      Very low (<0.3%): ~0.20 (possible engagement manipulation)
    """
    # Hard bot signal
    if avg_comments < 0.5 and engagement_rate < 0.003:
        return 0.12

    if avg_comments < 1:
        return 0.38   # Few comments → uncertain, use cautiously neutral value

    if engagement_rate > 0.05:
        return 0.80
    elif engagement_rate > 0.02:
        return 0.68
    elif engagement_rate > 0.01:
        return 0.55
    elif engagement_rate > 0.003:
        return 0.40
    else:
        return 0.22
