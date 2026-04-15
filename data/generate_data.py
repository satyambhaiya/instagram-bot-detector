"""
Instagram Bot Detection — Synthetic Data Generator
Generates realistic Instagram account profiles for 3 classes:
  0 = Human
  1 = Bot
  2 = Suspicious
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os

np.random.seed(42)

# ── Label map ──────────────────────────────────────────────────────────────
LABELS = {0: "Human", 1: "Bot", 2: "Suspicious"}

def _lognorm(mu, sigma, lo=0, hi=None):
    v = np.random.lognormal(mu, sigma)
    if hi: v = min(v, hi)
    return max(v, lo)

def _norm(mu, sigma, lo=0.0, hi=1.0):
    return float(np.clip(np.random.normal(mu, sigma), lo, hi))

def _beta(a, b, scale=1.0):
    return float(np.random.beta(a, b) * scale)

# ── Per-class generators ───────────────────────────────────────────────────

def gen_human():
    """
    Real-world calibrated human Instagram account.

    Key corrections vs v1:
    - Engagement rate lowered to 1-8% (real Instagram average is 1-5%, not 3-12%)
    - comment_reply_rate lowered to 0.25-0.65 (real accounts don't reply to 80% of comments)
    - unique_commenters_ratio lowered to 0.50-0.85 (real community, not 90%+ unique)
    - story_frequency kept at ~1/day but with more variance
    - posts_per_day adjusted to real posting cadence (0.2-1.5/day is typical)
    """
    followers   = int(_lognorm(7.2, 1.4, 50, 500_000))
    following   = int(_lognorm(5.8, 0.9, 10, 3_000))
    posts       = int(_lognorm(4.0, 1.2, 1, 2000))
    avg_likes   = int(followers * _beta(2, 8, 0.08))    # 1-8% ER (was 3-12%)
    avg_comments= int(avg_likes * _beta(1, 8, 0.10))
    posts_day   = _norm(0.4, 0.35, 0.01, 3.0)           # realistic posting cadence
    return {
        "followers_count":       followers,
        "following_count":       following,
        "posts_count":           posts,
        "avg_likes_per_post":    avg_likes,
        "avg_comments_per_post": avg_comments,
        "posts_per_day":         round(posts_day, 3),
        "follower_following_ratio": round(followers / max(following, 1), 4),
        "engagement_rate":       round(avg_likes / max(followers, 1), 4),
        "profile_pic":           int(np.random.choice([0,1], p=[0.04, 0.96])),
        "bio_length":            int(np.clip(np.random.normal(68, 35), 0, 150)),
        "has_url_in_bio":        int(np.random.choice([0,1], p=[0.48, 0.52])),
        "is_verified":           int(np.random.choice([0,1], p=[0.94, 0.06])),
        "is_private":            int(np.random.choice([0,1], p=[0.45, 0.55])),
        "account_age_days":      int(np.random.uniform(180, 4000)),
        "username_digit_ratio":  round(_beta(1, 8, 0.25), 3),
        "username_length":       int(np.clip(np.random.normal(11, 3), 4, 28)),
        "night_activity_ratio":  round(_beta(2, 6, 0.5), 3),
        "avg_caption_length":    int(np.clip(np.random.normal(90, 50), 0, 300)),
        "hashtags_per_post":     round(_norm(5.5, 3.5, 0, 30), 1),
        "mentions_per_post":     round(_norm(0.8, 1.0, 0, 10), 1),
        "story_frequency":       round(_norm(1.1, 0.8, 0.1, 4.0), 2),  # 0.3-2.5 typical
        "reels_ratio":           round(_beta(3, 4), 3),
        "comment_reply_rate":    round(_beta(3, 4, 0.65), 3),   # 0.25-0.65 real range
        "unique_commenters_ratio": round(_beta(5, 3, 0.82), 3), # 0.50-0.85 real range
        # v2.1 — engagement quality + behavioral consistency + mass-follow
        "likes_comments_ratio":  round(min(avg_likes / max(avg_comments, 0.1), 100.0), 4),
        "posting_regularity":    round(_beta(5, 2, 0.9), 3),    # 0.5-0.9 (irregular/human)
        "following_to_followers_ratio": round(min(following / max(followers, 1), 100.0), 4),
        "label": 0
    }

def gen_bot():
    """Generate a bot account. Randomly picks from 3 subtypes to ensure
    the model learns to detect diverse bot patterns."""
    subtype = np.random.choice(["spam", "popular_auto", "content_gen"],
                               p=[0.50, 0.25, 0.25])
    if subtype == "popular_auto":
        return _gen_bot_popular_automation()
    if subtype == "content_gen":
        return _gen_bot_content_generator()
    return _gen_bot_spam()


def _gen_bot_spam():
    """Classic spam bot: mass-follows, low engagement, lots of hashtags."""
    followers   = int(_lognorm(5.5, 2.0, 0, 100_000))
    following   = int(_lognorm(8.2, 1.2, 500, 7_500))
    posts       = int(_lognorm(2.8, 1.5, 0, 500))
    avg_likes   = int(followers * _beta(1, 20, 0.02))
    avg_comments= int(avg_likes * _beta(1, 15, 0.05))
    posts_day   = _norm(8.0, 5.0, 0.5, 50.0)
    return {
        "followers_count":       followers,
        "following_count":       following,
        "posts_count":           posts,
        "avg_likes_per_post":    avg_likes,
        "avg_comments_per_post": avg_comments,
        "posts_per_day":         round(posts_day, 3),
        "follower_following_ratio": round(followers / max(following, 1), 4),
        "engagement_rate":       round(avg_likes / max(followers, 1), 4),
        "profile_pic":           int(np.random.choice([0,1], p=[0.42, 0.58])),
        "bio_length":            int(np.clip(np.random.normal(12, 18), 0, 100)),
        "has_url_in_bio":        int(np.random.choice([0,1], p=[0.35, 0.65])),
        "is_verified":           int(np.random.choice([0,1], p=[0.995, 0.005])),
        "is_private":            int(np.random.choice([0,1], p=[0.85, 0.15])),
        "account_age_days":      int(np.random.uniform(1, 400)),
        "username_digit_ratio":  round(_beta(5, 3, 0.7), 3),
        "username_length":       int(np.clip(np.random.normal(16, 4), 6, 28)),
        "night_activity_ratio":  round(_beta(5, 2, 0.8), 3),
        "avg_caption_length":    int(np.clip(np.random.normal(15, 20), 0, 150)),
        "hashtags_per_post":     round(_norm(22.0, 5.0, 0, 30), 1),
        "mentions_per_post":     round(_norm(4.5, 3.0, 0, 15), 1),
        "story_frequency":       round(_norm(0.1, 0.2, 0, 1), 2),
        "reels_ratio":           round(_beta(1, 8), 3),
        "comment_reply_rate":    round(_beta(1, 9, 0.15), 3),
        "unique_commenters_ratio": round(_beta(1, 6, 0.3), 3),
        # v2.1 — spam bots: very high following/followers, low regularity
        "likes_comments_ratio":  round(min(avg_likes / max(avg_comments, 0.1), 100.0), 4),
        "posting_regularity":    round(_beta(2, 8, 0.3), 3),    # 0.02-0.15 (very regular/bot)
        "following_to_followers_ratio": round(min(following / max(followers, 1), 100.0), 4),
        "label": 1
    }


def _gen_bot_popular_automation():
    """Popular automation bot: high followers, near-zero following,
    posts at all hours, very high posting rate. E.g. @big_ben_clock."""
    followers   = int(_lognorm(10.0, 1.5, 5_000, 2_000_000))
    following   = int(np.random.choice([0, 0, 0, 1, 2, 3, 5]))  # almost zero
    posts       = int(_lognorm(7.0, 1.5, 500, 200_000))
    avg_likes   = int(followers * _beta(2, 5, 0.05))  # moderate ER (popular)
    avg_comments= int(avg_likes * _beta(1, 6, 0.04))
    posts_day   = _norm(12.0, 6.0, 4.0, 50.0)         # very high
    return {
        "followers_count":       followers,
        "following_count":       following,
        "posts_count":           posts,
        "avg_likes_per_post":    avg_likes,
        "avg_comments_per_post": avg_comments,
        "posts_per_day":         round(posts_day, 3),
        "follower_following_ratio": min(round(followers / max(following, 1), 4), 500.0),
        "engagement_rate":       round(avg_likes / max(followers, 1), 4),
        "profile_pic":           1,                      # always has pic
        "bio_length":            int(np.clip(np.random.normal(90, 40), 20, 160)),
        "has_url_in_bio":        int(np.random.choice([0,1], p=[0.50, 0.50])),
        "is_verified":           0,
        "is_private":            0,
        "account_age_days":      int(np.random.uniform(500, 5500)),  # old
        "username_digit_ratio":  round(_beta(1, 6, 0.15), 3),
        "username_length":       int(np.clip(np.random.normal(13, 3), 6, 22)),
        "night_activity_ratio":  round(_beta(5, 3, 0.7), 3),  # very high (24/7)
        "avg_caption_length":    int(np.clip(np.random.normal(40, 30), 5, 120)),
        "hashtags_per_post":     round(_norm(0.5, 1.0, 0, 5), 1),   # few/no hashtags
        "mentions_per_post":     round(_norm(0.3, 0.5, 0, 3), 1),   # few/no mentions
        "story_frequency":       round(_norm(0.05, 0.1, 0, 0.3), 2),  # no stories
        "reels_ratio":           round(_beta(1, 10), 3),
        "comment_reply_rate":    round(_beta(1, 15, 0.05), 3),  # never replies
        "unique_commenters_ratio": round(_beta(1, 5, 0.25), 3),  # low diversity
        # v2.1 — popular auto: very regular, near-zero following/followers
        "likes_comments_ratio":  round(min(avg_likes / max(avg_comments, 0.1), 100.0), 4),
        "posting_regularity":    round(_beta(1, 10, 0.15), 3),  # 0.01-0.08 (extremely regular)
        "following_to_followers_ratio": round(min(following / max(followers, 1), 100.0), 4),
        "label": 1
    }


def _gen_bot_content_generator():
    """Content generator bot: moderate-high followers, very few following,
    regular high-frequency posting, short repetitive content.
    E.g. @EmojiMashupBot, @year_progress."""
    followers   = int(_lognorm(9.0, 1.8, 1_000, 1_000_000))
    following   = int(np.clip(np.random.exponential(3), 0, 20))
    posts       = int(_lognorm(6.5, 1.5, 500, 100_000))
    avg_likes   = int(followers * _beta(2, 4, 0.08))  # good engagement
    avg_comments= int(avg_likes * _beta(1, 5, 0.03))
    posts_day   = _norm(6.0, 4.0, 1.5, 30.0)
    return {
        "followers_count":       followers,
        "following_count":       following,
        "posts_count":           posts,
        "avg_likes_per_post":    avg_likes,
        "avg_comments_per_post": avg_comments,
        "posts_per_day":         round(posts_day, 3),
        "follower_following_ratio": min(round(followers / max(following, 1), 4), 500.0),
        "engagement_rate":       round(avg_likes / max(followers, 1), 4),
        "profile_pic":           1,
        "bio_length":            int(np.clip(np.random.normal(100, 40), 30, 160)),
        "has_url_in_bio":        int(np.random.choice([0,1], p=[0.55, 0.45])),
        "is_verified":           0,
        "is_private":            0,
        "account_age_days":      int(np.random.uniform(300, 4000)),
        "username_digit_ratio":  round(_beta(1, 8, 0.10), 3),
        "username_length":       int(np.clip(np.random.normal(14, 4), 6, 25)),
        "night_activity_ratio":  round(_beta(3, 5, 0.45), 3),  # moderate night
        "avg_caption_length":    int(np.clip(np.random.normal(35, 25), 5, 100)),
        "hashtags_per_post":     round(_norm(0.8, 1.2, 0, 5), 1),
        "mentions_per_post":     round(_norm(0.4, 0.8, 0, 3), 1),
        "story_frequency":       round(_norm(0.08, 0.15, 0, 0.5), 2),
        "reels_ratio":           round(_beta(2, 6), 3),
        "comment_reply_rate":    round(_beta(1, 12, 0.08), 3),  # rarely replies
        "unique_commenters_ratio": round(_beta(2, 5, 0.35), 3),
        # v2.1 — content gen: regular posting, low following/followers
        "likes_comments_ratio":  round(min(avg_likes / max(avg_comments, 0.1), 100.0), 4),
        "posting_regularity":    round(_beta(1, 8, 0.20), 3),   # 0.02-0.10 (very regular)
        "following_to_followers_ratio": round(min(following / max(followers, 1), 100.0), 4),
        "label": 1
    }

def gen_suspicious():
    """
    Suspicious/grey-zone account — may use automation partially or buy engagement.
    These accounts look partially human but have bot-like signals mixed in.
    Examples: growth-hacking accounts, engagement pod users, semi-automated accounts.
    """
    followers   = int(_lognorm(6.5, 1.8, 20, 200_000))
    following   = int(_lognorm(7.0, 1.3, 100, 5_000))
    posts       = int(_lognorm(3.5, 1.3, 1, 800))
    avg_likes   = int(followers * _beta(2, 10, 0.06))   # low-ish ER (possibly bought)
    avg_comments= int(avg_likes * _beta(1, 10, 0.06))
    posts_day   = _norm(2.5, 2.0, 0.1, 15.0)            # above average posting rate
    return {
        "followers_count":       followers,
        "following_count":       following,
        "posts_count":           posts,
        "avg_likes_per_post":    avg_likes,
        "avg_comments_per_post": avg_comments,
        "posts_per_day":         round(posts_day, 3),
        "follower_following_ratio": round(followers / max(following, 1), 4),
        "engagement_rate":       round(avg_likes / max(followers, 1), 4),
        "profile_pic":           int(np.random.choice([0,1], p=[0.18, 0.82])),
        "bio_length":            int(np.clip(np.random.normal(35, 30), 0, 130)),
        "has_url_in_bio":        int(np.random.choice([0,1], p=[0.40, 0.60])),
        "is_verified":           int(np.random.choice([0,1], p=[0.97, 0.03])),
        "is_private":            int(np.random.choice([0,1], p=[0.60, 0.40])),
        "account_age_days":      int(np.random.uniform(60, 1500)),
        "username_digit_ratio":  round(_beta(3, 5, 0.45), 3),
        "username_length":       int(np.clip(np.random.normal(13, 4), 4, 28)),
        "night_activity_ratio":  round(_beta(3, 4, 0.6), 3),
        "avg_caption_length":    int(np.clip(np.random.normal(45, 35), 0, 200)),
        "hashtags_per_post":     round(_norm(13.0, 6.0, 0, 30), 1),
        "mentions_per_post":     round(_norm(2.2, 1.8, 0, 10), 1),
        "story_frequency":       round(_norm(0.45, 0.4, 0.0, 2.0), 2),
        "reels_ratio":           round(_beta(2, 5), 3),
        "comment_reply_rate":    round(_beta(2, 6, 0.35), 3),   # lower than human
        "unique_commenters_ratio": round(_beta(3, 6, 0.50), 3), # lower than human
        # v2.1 — suspicious: moderate regularity, slightly high following/followers
        "likes_comments_ratio":  round(min(avg_likes / max(avg_comments, 0.1), 100.0), 4),
        "posting_regularity":    round(_beta(3, 4, 0.6), 3),    # 0.2-0.5 (semi-regular)
        "following_to_followers_ratio": round(min(following / max(followers, 1), 100.0), 4),
        "label": 2
    }


def generate_dataset(n=12_000):
    n_human = int(n * 0.50)
    n_bot   = int(n * 0.30)
    n_susp  = n - n_human - n_bot

    rows = (
        [gen_human()      for _ in range(n_human)] +
        [gen_bot()        for _ in range(n_bot)]   +
        [gen_suspicious() for _ in range(n_susp)]
    )
    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def save_splits(df, out_dir="."):
    os.makedirs(out_dir, exist_ok=True)
    train, tmp = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=42)
    val, test  = train_test_split(tmp, test_size=0.50, stratify=tmp["label"], random_state=42)
    train.to_csv(f"{out_dir}/train.csv", index=False)
    val.to_csv(f"{out_dir}/val.csv",     index=False)
    test.to_csv(f"{out_dir}/test.csv",   index=False)
    print(f"✅  {len(df):,} samples  → train:{len(train):,} val:{len(val):,} test:{len(test):,}")
    vc = df["label"].value_counts().sort_index()
    for k,v in vc.items():
        print(f"    {LABELS[k]:>12}: {v:,}  ({v/len(df):.1%})")
    return train, val, test


if __name__ == "__main__":
    df = generate_dataset(12_000)
    save_splits(df, out_dir=".")
    print(df.describe().T[["mean","std","min","max"]].round(2).to_string())
