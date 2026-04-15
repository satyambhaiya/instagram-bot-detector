"""
Multi-Platform Synthetic Data Generator — calibrated on research literature.

Generates realistic social media profiles for bot detection training
across Twitter/X, Facebook, and Snapchat.

Calibration sources:
  - Twitter: Cresci et al. 2017, Varol et al. 2017, Ferrara 2020
  - Facebook: Boshmaf et al. 2011, Facebook Transparency Reports
  - Snapchat: Platform-specific patterns from security research

The 24 features are universal across platforms. Values are calibrated
per-platform based on documented behavioral differences.

Usage:
  python data/generate_platform_data.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

np.random.seed(42)

LABELS = {0: "Human", 1: "Bot", 2: "Suspicious"}

ROOT = Path(__file__).resolve().parent.parent


def _lognorm(mu, sigma, lo=0, hi=None):
    v = np.random.lognormal(mu, sigma)
    if hi: v = min(v, hi)
    return max(v, lo)

def _norm(mu, sigma, lo=0.0, hi=1.0):
    return float(np.clip(np.random.normal(mu, sigma), lo, hi))

def _beta(a, b, scale=1.0):
    return float(np.random.beta(a, b) * scale)


# ═══════════════════════════════════════════════════════════════════════════════
# TWITTER/X DATA GENERATORS
# Calibrated on Cresci-2017, Varol-2017, Twitter research literature
# ═══════════════════════════════════════════════════════════════════════════════

def gen_twitter_human():
    """Real human Twitter account — calibrated on Cresci-2017 genuine accounts."""
    followers   = int(_lognorm(5.8, 2.0, 5, 1_000_000))
    following   = int(_lognorm(5.2, 1.5, 1, 5_000))
    tweets      = int(_lognorm(6.5, 2.0, 1, 100_000))
    # Twitter engagement: typically 0.5-3% for organic accounts
    avg_likes   = int(followers * _beta(2, 15, 0.03))
    avg_replies = int(avg_likes * _beta(1, 10, 0.08))
    # Twitter users tweet 1-5x/day on average
    tweets_day  = _norm(1.5, 2.0, 0.01, 10.0)
    return {
        "followers_count":       followers,
        "following_count":       following,
        "posts_count":           tweets,
        "avg_likes_per_post":    avg_likes,
        "avg_comments_per_post": avg_replies,
        "posts_per_day":         round(tweets_day, 3),
        "follower_following_ratio": round(followers / max(following, 1), 4),
        "engagement_rate":       round(avg_likes / max(followers, 1), 4),
        "profile_pic":           int(np.random.choice([0,1], p=[0.06, 0.94])),
        "bio_length":            int(np.clip(np.random.normal(80, 45), 0, 160)),
        "has_url_in_bio":        int(np.random.choice([0,1], p=[0.55, 0.45])),
        "is_verified":           int(np.random.choice([0,1], p=[0.97, 0.03])),
        "is_private":            int(np.random.choice([0,1], p=[0.85, 0.15])),
        "account_age_days":      int(np.random.uniform(180, 5000)),
        "username_digit_ratio":  round(_beta(1.5, 8, 0.3), 3),
        "username_length":       int(np.clip(np.random.normal(10, 4), 3, 15)),
        "night_activity_ratio":  round(_beta(2, 5, 0.5), 3),
        "avg_caption_length":    int(np.clip(np.random.normal(100, 60), 5, 280)),
        "hashtags_per_post":     round(_norm(1.5, 1.5, 0, 10), 1),
        "mentions_per_post":     round(_norm(1.2, 1.0, 0, 8), 1),
        "story_frequency":       round(_norm(0.3, 0.4, 0.0, 2.0), 2),  # Twitter Fleets/Spaces
        "reels_ratio":           round(_beta(1, 6), 3),  # video tweets
        "comment_reply_rate":    round(_beta(3, 4, 0.55), 3),
        "unique_commenters_ratio": round(_beta(5, 3, 0.78), 3),
        "likes_comments_ratio":  round(min(avg_likes / max(avg_replies, 0.1), 100.0), 4),
        "posting_regularity":    round(_beta(5, 2, 0.85), 3),
        "following_to_followers_ratio": round(min(following / max(followers, 1), 100.0), 4),
        "label": 0
    }

def gen_twitter_bot():
    """Twitter bot — randomly picks from spam, popular automation, or content gen."""
    subtype = np.random.choice(["spam", "popular_auto", "content_gen"],
                               p=[0.45, 0.30, 0.25])
    if subtype == "popular_auto":
        return _gen_twitter_bot_popular()
    if subtype == "content_gen":
        return _gen_twitter_bot_content()
    return _gen_twitter_bot_spam()


def _gen_twitter_bot_spam():
    """Classic spam bot: mass-follows, low engagement, lots of hashtags."""
    followers   = int(_lognorm(3.5, 2.5, 0, 50_000))
    following   = int(_lognorm(7.5, 1.5, 100, 10_000))
    tweets      = int(_lognorm(5.0, 2.5, 0, 500_000))
    avg_likes   = int(followers * _beta(1, 30, 0.01))
    avg_replies = int(avg_likes * _beta(1, 20, 0.03))
    tweets_day  = _norm(20.0, 15.0, 0.5, 100.0)
    return {
        "followers_count":       followers,
        "following_count":       following,
        "posts_count":           tweets,
        "avg_likes_per_post":    avg_likes,
        "avg_comments_per_post": avg_replies,
        "posts_per_day":         round(tweets_day, 3),
        "follower_following_ratio": round(followers / max(following, 1), 4),
        "engagement_rate":       round(avg_likes / max(followers, 1), 4),
        "profile_pic":           int(np.random.choice([0,1], p=[0.35, 0.65])),
        "bio_length":            int(np.clip(np.random.normal(15, 20), 0, 160)),
        "has_url_in_bio":        int(np.random.choice([0,1], p=[0.30, 0.70])),
        "is_verified":           int(np.random.choice([0,1], p=[0.998, 0.002])),
        "is_private":            int(np.random.choice([0,1], p=[0.92, 0.08])),
        "account_age_days":      int(np.random.uniform(1, 300)),
        "username_digit_ratio":  round(_beta(5, 2, 0.8), 3),
        "username_length":       int(np.clip(np.random.normal(14, 3), 5, 15)),
        "night_activity_ratio":  round(_beta(5, 2, 0.85), 3),
        "avg_caption_length":    int(np.clip(np.random.normal(30, 30), 0, 280)),
        "hashtags_per_post":     round(_norm(6.0, 4.0, 0, 20), 1),
        "mentions_per_post":     round(_norm(5.0, 3.0, 0, 15), 1),
        "story_frequency":       round(_norm(0.02, 0.05, 0.0, 0.3), 2),
        "reels_ratio":           round(_beta(1, 10), 3),
        "comment_reply_rate":    round(_beta(1, 9, 0.12), 3),
        "unique_commenters_ratio": round(_beta(1, 8, 0.2), 3),
        "likes_comments_ratio":  round(min(avg_likes / max(avg_replies, 0.1), 100.0), 4),
        "posting_regularity":    round(_beta(2, 8, 0.25), 3),
        "following_to_followers_ratio": round(min(following / max(followers, 1), 100.0), 4),
        "label": 1
    }


def _gen_twitter_bot_popular():
    """Popular Twitter automation bot — e.g. @big_ben_clock, @earthquakeBot."""
    followers   = int(_lognorm(10.5, 1.8, 10_000, 2_000_000))
    following   = int(np.random.choice([0, 0, 0, 0, 1, 2, 5]))
    tweets      = int(_lognorm(8.0, 1.5, 5_000, 500_000))
    avg_likes   = int(followers * _beta(2, 6, 0.06))
    avg_replies = int(avg_likes * _beta(1, 8, 0.03))
    tweets_day  = _norm(15.0, 8.0, 3.0, 48.0)
    return {
        "followers_count":       followers,
        "following_count":       following,
        "posts_count":           tweets,
        "avg_likes_per_post":    avg_likes,
        "avg_comments_per_post": avg_replies,
        "posts_per_day":         round(tweets_day, 3),
        "follower_following_ratio": min(round(followers / max(following, 1), 4), 500.0),
        "engagement_rate":       round(avg_likes / max(followers, 1), 4),
        "profile_pic":           1,
        "bio_length":            int(np.clip(np.random.normal(100, 40), 20, 160)),
        "has_url_in_bio":        int(np.random.choice([0,1], p=[0.45, 0.55])),
        "is_verified":           0,
        "is_private":            0,
        "account_age_days":      int(np.random.uniform(800, 6000)),
        "username_digit_ratio":  round(_beta(1, 6, 0.15), 3),
        "username_length":       int(np.clip(np.random.normal(13, 3), 6, 22)),
        "night_activity_ratio":  round(_beta(5, 3, 0.75), 3),
        "avg_caption_length":    int(np.clip(np.random.normal(50, 30), 5, 200)),
        "hashtags_per_post":     round(_norm(0.3, 0.8, 0, 3), 1),
        "mentions_per_post":     round(_norm(0.2, 0.5, 0, 2), 1),
        "story_frequency":       round(_norm(0.02, 0.05, 0.0, 0.2), 2),
        "reels_ratio":           round(_beta(1, 12), 3),
        "comment_reply_rate":    round(_beta(1, 15, 0.04), 3),
        "unique_commenters_ratio": round(_beta(1, 6, 0.2), 3),
        "likes_comments_ratio":  round(min(avg_likes / max(avg_replies, 0.1), 100.0), 4),
        "posting_regularity":    round(_beta(1, 10, 0.12), 3),
        "following_to_followers_ratio": round(min(following / max(followers, 1), 100.0), 4),
        "label": 1
    }


def _gen_twitter_bot_content():
    """Content generator bot — posts automated content regularly."""
    followers   = int(_lognorm(8.5, 2.0, 500, 500_000))
    following   = int(np.clip(np.random.exponential(5), 0, 30))
    tweets      = int(_lognorm(7.0, 1.5, 1_000, 100_000))
    avg_likes   = int(followers * _beta(2, 5, 0.08))
    avg_replies = int(avg_likes * _beta(1, 6, 0.03))
    tweets_day  = _norm(8.0, 5.0, 1.0, 30.0)
    return {
        "followers_count":       followers,
        "following_count":       following,
        "posts_count":           tweets,
        "avg_likes_per_post":    avg_likes,
        "avg_comments_per_post": avg_replies,
        "posts_per_day":         round(tweets_day, 3),
        "follower_following_ratio": min(round(followers / max(following, 1), 4), 500.0),
        "engagement_rate":       round(avg_likes / max(followers, 1), 4),
        "profile_pic":           1,
        "bio_length":            int(np.clip(np.random.normal(110, 40), 30, 160)),
        "has_url_in_bio":        int(np.random.choice([0,1], p=[0.50, 0.50])),
        "is_verified":           0,
        "is_private":            0,
        "account_age_days":      int(np.random.uniform(400, 4500)),
        "username_digit_ratio":  round(_beta(1, 8, 0.10), 3),
        "username_length":       int(np.clip(np.random.normal(14, 4), 6, 25)),
        "night_activity_ratio":  round(_beta(3, 5, 0.50), 3),
        "avg_caption_length":    int(np.clip(np.random.normal(45, 25), 5, 140)),
        "hashtags_per_post":     round(_norm(0.5, 1.0, 0, 4), 1),
        "mentions_per_post":     round(_norm(0.3, 0.7, 0, 3), 1),
        "story_frequency":       round(_norm(0.05, 0.1, 0.0, 0.3), 2),
        "reels_ratio":           round(_beta(1, 8), 3),
        "comment_reply_rate":    round(_beta(1, 12, 0.06), 3),
        "unique_commenters_ratio": round(_beta(2, 6, 0.30), 3),
        "likes_comments_ratio":  round(min(avg_likes / max(avg_replies, 0.1), 100.0), 4),
        "posting_regularity":    round(_beta(1, 8, 0.18), 3),
        "following_to_followers_ratio": round(min(following / max(followers, 1), 100.0), 4),
        "label": 1
    }

def gen_twitter_suspicious():
    """Suspicious Twitter account — partially automated, uses bots for growth."""
    followers   = int(_lognorm(5.0, 2.0, 10, 200_000))
    following   = int(_lognorm(6.5, 1.5, 50, 5_000))
    tweets      = int(_lognorm(6.0, 2.0, 10, 50_000))
    avg_likes   = int(followers * _beta(2, 12, 0.04))
    avg_replies = int(avg_likes * _beta(1, 12, 0.05))
    tweets_day  = _norm(5.0, 4.0, 0.1, 30.0)
    return {
        "followers_count":       followers,
        "following_count":       following,
        "posts_count":           tweets,
        "avg_likes_per_post":    avg_likes,
        "avg_comments_per_post": avg_replies,
        "posts_per_day":         round(tweets_day, 3),
        "follower_following_ratio": round(followers / max(following, 1), 4),
        "engagement_rate":       round(avg_likes / max(followers, 1), 4),
        "profile_pic":           int(np.random.choice([0,1], p=[0.15, 0.85])),
        "bio_length":            int(np.clip(np.random.normal(50, 35), 0, 160)),
        "has_url_in_bio":        int(np.random.choice([0,1], p=[0.35, 0.65])),
        "is_verified":           int(np.random.choice([0,1], p=[0.98, 0.02])),
        "is_private":            int(np.random.choice([0,1], p=[0.88, 0.12])),
        "account_age_days":      int(np.random.uniform(60, 1500)),
        "username_digit_ratio":  round(_beta(3, 5, 0.5), 3),
        "username_length":       int(np.clip(np.random.normal(12, 3), 4, 15)),
        "night_activity_ratio":  round(_beta(3, 3, 0.6), 3),
        "avg_caption_length":    int(np.clip(np.random.normal(60, 40), 0, 280)),
        "hashtags_per_post":     round(_norm(4.0, 3.0, 0, 15), 1),
        "mentions_per_post":     round(_norm(3.0, 2.0, 0, 10), 1),
        "story_frequency":       round(_norm(0.15, 0.2, 0.0, 1.0), 2),
        "reels_ratio":           round(_beta(2, 6), 3),
        "comment_reply_rate":    round(_beta(2, 6, 0.30), 3),
        "unique_commenters_ratio": round(_beta(3, 5, 0.45), 3),
        "likes_comments_ratio":  round(min(avg_likes / max(avg_replies, 0.1), 100.0), 4),
        "posting_regularity":    round(_beta(3, 4, 0.55), 3),
        "following_to_followers_ratio": round(min(following / max(followers, 1), 100.0), 4),
        "label": 2
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FACEBOOK DATA GENERATORS
# Calibrated on Facebook Transparency Reports + Boshmaf et al.
# ═══════════════════════════════════════════════════════════════════════════════

def gen_facebook_human():
    followers   = int(_lognorm(5.5, 1.8, 10, 100_000))
    friends     = int(_lognorm(5.0, 1.0, 5, 5_000))
    posts       = int(_lognorm(4.5, 1.5, 1, 5_000))
    avg_likes   = int(followers * _beta(2, 6, 0.10))
    avg_comments= int(avg_likes * _beta(1, 6, 0.10))
    posts_day   = _norm(0.3, 0.3, 0.01, 3.0)
    return {
        "followers_count":       followers,
        "following_count":       friends,
        "posts_count":           posts,
        "avg_likes_per_post":    avg_likes,
        "avg_comments_per_post": avg_comments,
        "posts_per_day":         round(posts_day, 3),
        "follower_following_ratio": round(followers / max(friends, 1), 4),
        "engagement_rate":       round(avg_likes / max(followers, 1), 4),
        "profile_pic":           int(np.random.choice([0,1], p=[0.03, 0.97])),
        "bio_length":            int(np.clip(np.random.normal(40, 30), 0, 200)),
        "has_url_in_bio":        int(np.random.choice([0,1], p=[0.70, 0.30])),
        "is_verified":           int(np.random.choice([0,1], p=[0.96, 0.04])),
        "is_private":            int(np.random.choice([0,1], p=[0.35, 0.65])),
        "account_age_days":      int(np.random.uniform(365, 5500)),
        "username_digit_ratio":  round(_beta(1, 10, 0.15), 3),
        "username_length":       int(np.clip(np.random.normal(12, 4), 4, 30)),
        "night_activity_ratio":  round(_beta(2, 7, 0.4), 3),
        "avg_caption_length":    int(np.clip(np.random.normal(60, 45), 0, 500)),
        "hashtags_per_post":     round(_norm(1.0, 1.5, 0, 10), 1),
        "mentions_per_post":     round(_norm(0.5, 0.8, 0, 5), 1),
        "story_frequency":       round(_norm(0.5, 0.5, 0.0, 3.0), 2),
        "reels_ratio":           round(_beta(2, 6), 3),
        "comment_reply_rate":    round(_beta(4, 3, 0.70), 3),
        "unique_commenters_ratio": round(_beta(5, 2, 0.85), 3),
        "likes_comments_ratio":  round(min(avg_likes / max(avg_comments, 0.1), 100.0), 4),
        "posting_regularity":    round(_beta(5, 2, 0.88), 3),
        "following_to_followers_ratio": round(min(friends / max(followers, 1), 100.0), 4),
        "label": 0
    }

def gen_facebook_bot():
    followers   = int(_lognorm(6.0, 2.5, 0, 200_000))
    friends     = int(_lognorm(7.0, 1.5, 100, 5_000))
    posts       = int(_lognorm(3.0, 1.5, 0, 1_000))
    avg_likes   = int(followers * _beta(1, 25, 0.02))
    avg_comments= int(avg_likes * _beta(1, 20, 0.03))
    posts_day   = _norm(5.0, 4.0, 0.1, 30.0)
    return {
        "followers_count":       followers,
        "following_count":       friends,
        "posts_count":           posts,
        "avg_likes_per_post":    avg_likes,
        "avg_comments_per_post": avg_comments,
        "posts_per_day":         round(posts_day, 3),
        "follower_following_ratio": round(followers / max(friends, 1), 4),
        "engagement_rate":       round(avg_likes / max(followers, 1), 4),
        "profile_pic":           int(np.random.choice([0,1], p=[0.30, 0.70])),
        "bio_length":            int(np.clip(np.random.normal(10, 15), 0, 100)),
        "has_url_in_bio":        int(np.random.choice([0,1], p=[0.25, 0.75])),
        "is_verified":           int(np.random.choice([0,1], p=[0.998, 0.002])),
        "is_private":            int(np.random.choice([0,1], p=[0.80, 0.20])),
        "account_age_days":      int(np.random.uniform(1, 200)),
        "username_digit_ratio":  round(_beta(5, 3, 0.6), 3),
        "username_length":       int(np.clip(np.random.normal(15, 5), 5, 30)),
        "night_activity_ratio":  round(_beta(5, 2, 0.8), 3),
        "avg_caption_length":    int(np.clip(np.random.normal(15, 20), 0, 200)),
        "hashtags_per_post":     round(_norm(8.0, 5.0, 0, 25), 1),
        "mentions_per_post":     round(_norm(3.0, 2.5, 0, 10), 1),
        "story_frequency":       round(_norm(0.05, 0.1, 0.0, 0.5), 2),
        "reels_ratio":           round(_beta(1, 8), 3),
        "comment_reply_rate":    round(_beta(1, 10, 0.10), 3),
        "unique_commenters_ratio": round(_beta(1, 8, 0.15), 3),
        "likes_comments_ratio":  round(min(avg_likes / max(avg_comments, 0.1), 100.0), 4),
        "posting_regularity":    round(_beta(2, 8, 0.25), 3),
        "following_to_followers_ratio": round(min(friends / max(followers, 1), 100.0), 4),
        "label": 1
    }

def gen_facebook_suspicious():
    followers   = int(_lognorm(5.8, 2.0, 20, 150_000))
    friends     = int(_lognorm(6.5, 1.3, 50, 5_000))
    posts       = int(_lognorm(4.0, 1.5, 5, 2_000))
    avg_likes   = int(followers * _beta(2, 12, 0.05))
    avg_comments= int(avg_likes * _beta(1, 10, 0.05))
    posts_day   = _norm(1.5, 1.5, 0.05, 10.0)
    return {
        "followers_count":       followers,
        "following_count":       friends,
        "posts_count":           posts,
        "avg_likes_per_post":    avg_likes,
        "avg_comments_per_post": avg_comments,
        "posts_per_day":         round(posts_day, 3),
        "follower_following_ratio": round(followers / max(friends, 1), 4),
        "engagement_rate":       round(avg_likes / max(followers, 1), 4),
        "profile_pic":           int(np.random.choice([0,1], p=[0.12, 0.88])),
        "bio_length":            int(np.clip(np.random.normal(30, 25), 0, 150)),
        "has_url_in_bio":        int(np.random.choice([0,1], p=[0.40, 0.60])),
        "is_verified":           int(np.random.choice([0,1], p=[0.97, 0.03])),
        "is_private":            int(np.random.choice([0,1], p=[0.55, 0.45])),
        "account_age_days":      int(np.random.uniform(60, 1200)),
        "username_digit_ratio":  round(_beta(3, 5, 0.4), 3),
        "username_length":       int(np.clip(np.random.normal(13, 4), 4, 30)),
        "night_activity_ratio":  round(_beta(3, 4, 0.55), 3),
        "avg_caption_length":    int(np.clip(np.random.normal(35, 30), 0, 300)),
        "hashtags_per_post":     round(_norm(4.0, 3.5, 0, 15), 1),
        "mentions_per_post":     round(_norm(1.5, 1.5, 0, 8), 1),
        "story_frequency":       round(_norm(0.2, 0.25, 0.0, 1.5), 2),
        "reels_ratio":           round(_beta(2, 6), 3),
        "comment_reply_rate":    round(_beta(2, 5, 0.30), 3),
        "unique_commenters_ratio": round(_beta(3, 5, 0.40), 3),
        "likes_comments_ratio":  round(min(avg_likes / max(avg_comments, 0.1), 100.0), 4),
        "posting_regularity":    round(_beta(3, 4, 0.55), 3),
        "following_to_followers_ratio": round(min(friends / max(followers, 1), 100.0), 4),
        "label": 2
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SNAPCHAT DATA GENERATORS
# Limited public data — calibrated on Snapchat's platform characteristics
# ═══════════════════════════════════════════════════════════════════════════════

def gen_snapchat_human():
    followers   = int(_lognorm(4.5, 1.5, 5, 50_000))
    following   = int(_lognorm(4.3, 1.2, 5, 5_000))
    snap_score  = int(_lognorm(8.0, 2.0, 10, 1_000_000))
    avg_likes   = int(followers * _beta(3, 6, 0.12))
    avg_comments= int(avg_likes * _beta(1, 8, 0.06))
    snaps_day   = _norm(3.0, 3.0, 0.1, 20.0)
    return {
        "followers_count":       followers,
        "following_count":       following,
        "posts_count":           snap_score,
        "avg_likes_per_post":    avg_likes,
        "avg_comments_per_post": avg_comments,
        "posts_per_day":         round(snaps_day, 3),
        "follower_following_ratio": round(followers / max(following, 1), 4),
        "engagement_rate":       round(avg_likes / max(followers, 1), 4),
        "profile_pic":           int(np.random.choice([0,1], p=[0.08, 0.92])),
        "bio_length":            int(np.clip(np.random.normal(15, 15), 0, 80)),
        "has_url_in_bio":        int(np.random.choice([0,1], p=[0.85, 0.15])),
        "is_verified":           int(np.random.choice([0,1], p=[0.99, 0.01])),
        "is_private":            int(np.random.choice([0,1], p=[0.40, 0.60])),
        "account_age_days":      int(np.random.uniform(180, 3500)),
        "username_digit_ratio":  round(_beta(1.5, 7, 0.25), 3),
        "username_length":       int(np.clip(np.random.normal(9, 3), 3, 15)),
        "night_activity_ratio":  round(_beta(2, 5, 0.45), 3),
        "avg_caption_length":    int(np.clip(np.random.normal(20, 20), 0, 100)),
        "hashtags_per_post":     round(_norm(0.3, 0.5, 0, 5), 1),
        "mentions_per_post":     round(_norm(0.5, 0.8, 0, 5), 1),
        "story_frequency":       round(_norm(2.5, 2.0, 0.1, 10.0), 2),
        "reels_ratio":           round(_beta(1, 4), 3),
        "comment_reply_rate":    round(_beta(4, 3, 0.60), 3),
        "unique_commenters_ratio": round(_beta(5, 3, 0.75), 3),
        "likes_comments_ratio":  round(min(avg_likes / max(avg_comments, 0.1), 100.0), 4),
        "posting_regularity":    round(_beta(5, 2, 0.85), 3),
        "following_to_followers_ratio": round(min(following / max(followers, 1), 100.0), 4),
        "label": 0
    }

def gen_snapchat_bot():
    followers   = int(_lognorm(5.5, 2.5, 0, 100_000))
    following   = int(_lognorm(6.5, 1.5, 50, 10_000))
    snap_score  = int(_lognorm(4.0, 2.0, 0, 10_000))
    avg_likes   = int(followers * _beta(1, 25, 0.01))
    avg_comments= int(avg_likes * _beta(1, 20, 0.02))
    snaps_day   = _norm(0.5, 1.0, 0.0, 5.0)
    return {
        "followers_count":       followers,
        "following_count":       following,
        "posts_count":           snap_score,
        "avg_likes_per_post":    avg_likes,
        "avg_comments_per_post": avg_comments,
        "posts_per_day":         round(snaps_day, 3),
        "follower_following_ratio": round(followers / max(following, 1), 4),
        "engagement_rate":       round(avg_likes / max(followers, 1), 4),
        "profile_pic":           int(np.random.choice([0,1], p=[0.45, 0.55])),
        "bio_length":            int(np.clip(np.random.normal(5, 10), 0, 60)),
        "has_url_in_bio":        int(np.random.choice([0,1], p=[0.30, 0.70])),
        "is_verified":           int(np.random.choice([0,1], p=[0.999, 0.001])),
        "is_private":            int(np.random.choice([0,1], p=[0.75, 0.25])),
        "account_age_days":      int(np.random.uniform(1, 180)),
        "username_digit_ratio":  round(_beta(5, 2, 0.75), 3),
        "username_length":       int(np.clip(np.random.normal(13, 4), 5, 15)),
        "night_activity_ratio":  round(_beta(5, 2, 0.80), 3),
        "avg_caption_length":    int(np.clip(np.random.normal(5, 10), 0, 50)),
        "hashtags_per_post":     round(_norm(0.1, 0.3, 0, 3), 1),
        "mentions_per_post":     round(_norm(3.0, 2.5, 0, 10), 1),
        "story_frequency":       round(_norm(0.05, 0.1, 0.0, 0.5), 2),
        "reels_ratio":           round(_beta(1, 10), 3),
        "comment_reply_rate":    round(_beta(1, 10, 0.08), 3),
        "unique_commenters_ratio": round(_beta(1, 8, 0.12), 3),
        "likes_comments_ratio":  round(min(avg_likes / max(avg_comments, 0.1), 100.0), 4),
        "posting_regularity":    round(_beta(2, 8, 0.22), 3),
        "following_to_followers_ratio": round(min(following / max(followers, 1), 100.0), 4),
        "label": 1
    }

def gen_snapchat_suspicious():
    followers   = int(_lognorm(5.0, 2.0, 10, 80_000))
    following   = int(_lognorm(5.5, 1.5, 20, 5_000))
    snap_score  = int(_lognorm(6.0, 2.0, 100, 200_000))
    avg_likes   = int(followers * _beta(2, 10, 0.05))
    avg_comments= int(avg_likes * _beta(1, 10, 0.04))
    snaps_day   = _norm(1.0, 1.5, 0.05, 8.0)
    return {
        "followers_count":       followers,
        "following_count":       following,
        "posts_count":           snap_score,
        "avg_likes_per_post":    avg_likes,
        "avg_comments_per_post": avg_comments,
        "posts_per_day":         round(snaps_day, 3),
        "follower_following_ratio": round(followers / max(following, 1), 4),
        "engagement_rate":       round(avg_likes / max(followers, 1), 4),
        "profile_pic":           int(np.random.choice([0,1], p=[0.18, 0.82])),
        "bio_length":            int(np.clip(np.random.normal(12, 12), 0, 60)),
        "has_url_in_bio":        int(np.random.choice([0,1], p=[0.50, 0.50])),
        "is_verified":           int(np.random.choice([0,1], p=[0.995, 0.005])),
        "is_private":            int(np.random.choice([0,1], p=[0.55, 0.45])),
        "account_age_days":      int(np.random.uniform(30, 1000)),
        "username_digit_ratio":  round(_beta(3, 5, 0.45), 3),
        "username_length":       int(np.clip(np.random.normal(11, 3), 4, 15)),
        "night_activity_ratio":  round(_beta(3, 3, 0.55), 3),
        "avg_caption_length":    int(np.clip(np.random.normal(12, 15), 0, 80)),
        "hashtags_per_post":     round(_norm(0.5, 0.8, 0, 5), 1),
        "mentions_per_post":     round(_norm(1.5, 1.5, 0, 6), 1),
        "story_frequency":       round(_norm(0.8, 0.8, 0.0, 4.0), 2),
        "reels_ratio":           round(_beta(2, 6), 3),
        "comment_reply_rate":    round(_beta(2, 5, 0.25), 3),
        "unique_commenters_ratio": round(_beta(3, 5, 0.35), 3),
        "likes_comments_ratio":  round(min(avg_likes / max(avg_comments, 0.1), 100.0), 4),
        "posting_regularity":    round(_beta(3, 4, 0.50), 3),
        "following_to_followers_ratio": round(min(following / max(followers, 1), 100.0), 4),
        "label": 2
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

GENERATORS = {
    "twitter": {0: gen_twitter_human, 1: gen_twitter_bot, 2: gen_twitter_suspicious},
    "facebook": {0: gen_facebook_human, 1: gen_facebook_bot, 2: gen_facebook_suspicious},
    "snapchat": {0: gen_snapchat_human, 1: gen_snapchat_bot, 2: gen_snapchat_suspicious},
}


def generate_platform_dataset(platform: str, n: int = 10_000) -> pd.DataFrame:
    """Generate synthetic data for a specific platform."""
    gens = GENERATORS[platform]
    n_human = int(n * 0.50)
    n_bot   = int(n * 0.30)
    n_susp  = n - n_human - n_bot

    rows = (
        [gens[0]() for _ in range(n_human)] +
        [gens[1]() for _ in range(n_bot)]   +
        [gens[2]() for _ in range(n_susp)]
    )
    return pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)


def main():
    out_dir = Path(__file__).resolve().parent

    print("=" * 60)
    print("  Multi-Platform Data Generator")
    print("=" * 60)

    for platform in ["twitter", "facebook", "snapchat"]:
        print(f"\n── Generating {platform.upper()} data ──")
        df = generate_platform_dataset(platform, n=10_000)

        train, tmp = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=42)
        val, test  = train_test_split(tmp, test_size=0.50, stratify=tmp["label"], random_state=42)

        train.to_csv(out_dir / f"{platform}_train.csv", index=False)
        val.to_csv(out_dir / f"{platform}_val.csv", index=False)
        test.to_csv(out_dir / f"{platform}_test.csv", index=False)

        print(f"  {len(df):,} samples → train:{len(train):,} val:{len(val):,} test:{len(test):,}")
        for lid, lname in LABELS.items():
            cnt = (df["label"] == lid).sum()
            print(f"    {lname:>12}: {cnt:,}  ({cnt/len(df):.1%})")

    print(f"\n{'=' * 60}")
    print("  Done! Files saved to data/")
    print("=" * 60)


if __name__ == "__main__":
    main()
