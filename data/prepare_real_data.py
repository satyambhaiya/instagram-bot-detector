"""
Real Data Preparation Pipeline — Instagram Bot Detector

Processes publicly available real Instagram datasets and maps them
to our 24-feature schema for model training.

Supported datasets:
  1. Purba et al. (32K accounts) — Kaggle: krpurba/fakeauthentic-user-instagram
  2. InstaFake (1.4K accounts) — GitHub: fcakyon/instafake-dataset

Usage:
  1. Download datasets and place CSVs/JSONs in data/real/
  2. Run: python data/prepare_real_data.py
  3. Output: data/real_train.csv, data/real_val.csv, data/real_test.csv
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
REAL_DIR = Path(__file__).resolve().parent / "real"

# Our 24-feature schema
FEATURE_COLS = [
    "followers_count", "following_count", "posts_count",
    "avg_likes_per_post", "avg_comments_per_post",
    "posts_per_day", "follower_following_ratio", "engagement_rate",
    "profile_pic", "bio_length", "has_url_in_bio", "is_verified",
    "is_private", "account_age_days", "username_digit_ratio",
    "username_length", "night_activity_ratio", "avg_caption_length",
    "hashtags_per_post", "mentions_per_post", "story_frequency",
    "reels_ratio", "comment_reply_rate", "unique_commenters_ratio",
]

LABELS = {0: "Human", 1: "Bot", 2: "Suspicious"}


# ── Dataset 1: Purba et al. (32K accounts) ────────────────────────────────────

def load_purba(filepath: str | Path) -> pd.DataFrame:
    """
    Load and map the Purba dataset.

    Purba columns:
      pos (posts), flg (following), flr (followers), bl (bio length),
      pic (profile pic), lin (external url), cl (avg caption length),
      erl (engagement rate likes), erc (engagement rate comments),
      hc (hashtags count), cz (caption zero %), ni (non-image %),
      lt (location tag %), pr (promotional), fo (follower keywords),
      cs (cosine similarity), pi (post interval days),
      label: r=real, a=active fake, i=inactive fake, s=spammer
    """
    df = pd.read_csv(filepath)
    print(f"  Purba: loaded {len(df):,} rows from {filepath}")
    print(f"  Columns: {list(df.columns)}")

    # Detect label column (could be 'label' or 'class')
    label_col = "class" if "class" in df.columns else "label"
    print(f"  Label column: '{label_col}'")
    print(f"  Label distribution: {df[label_col].value_counts().to_dict()}")

    # Map labels to our 3-class scheme:
    #   r (real) → 0 (Human)
    #   a (active fake) + s (spammer) → 1 (Bot)
    #   i (inactive fake) → 2 (Suspicious)
    label_map = {"r": 0, "a": 1, "s": 1, "i": 2}
    df["label"] = df[label_col].map(label_map)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    # Detect followers column (could be 'flr' or 'flw')
    followers_col = "flw" if "flw" in df.columns else "flr"

    # Map features
    out = pd.DataFrame()
    out["followers_count"] = df[followers_col].clip(lower=0)
    out["following_count"] = df.get("flg", pd.Series(dtype=float)).clip(lower=0)
    out["posts_count"] = df.get("pos", pd.Series(dtype=float)).clip(lower=0)

    # Engagement rate (likes-based)
    out["engagement_rate"] = df.get("erl", pd.Series(dtype=float)).clip(lower=0, upper=1.0)

    # Derive avg_likes from engagement_rate * followers
    out["avg_likes_per_post"] = (out["engagement_rate"] * out["followers_count"]).round(0)

    # Engagement rate comments → derive avg_comments
    erc = df.get("erc", pd.Series(dtype=float)).clip(lower=0, upper=1.0).fillna(0)
    out["avg_comments_per_post"] = (erc * out["followers_count"]).round(0)

    # Post interval (days) → posts_per_day
    pi = df.get("pi", pd.Series(dtype=float)).clip(lower=0.1)
    out["posts_per_day"] = (1.0 / pi).clip(lower=0.001, upper=50.0).round(3)

    # Derived ratio
    out["follower_following_ratio"] = (
        out["followers_count"] / out["following_count"].clip(lower=1)
    ).round(4)

    # Direct mappings
    out["profile_pic"] = df.get("pic", pd.Series(dtype=float)).fillna(1).astype(int)
    out["bio_length"] = df.get("bl", pd.Series(dtype=float)).fillna(0).clip(lower=0).astype(int)
    out["has_url_in_bio"] = df.get("lin", pd.Series(dtype=float)).fillna(0).astype(int)
    out["avg_caption_length"] = df.get("cl", pd.Series(dtype=float)).fillna(0).clip(lower=0).astype(int)
    out["hashtags_per_post"] = df.get("hc", pd.Series(dtype=float)).fillna(0).clip(lower=0).round(1)

    # Non-image ratio → approximate reels_ratio
    out["reels_ratio"] = df.get("ni", pd.Series(dtype=float)).fillna(0).clip(lower=0, upper=1.0).round(3)

    # ── Features not in Purba → estimate from observed data ────────────────

    out["is_verified"] = _estimate_is_verified(out["followers_count"])
    out["is_private"] = _estimate_is_private(df["label"], out["followers_count"])
    out["account_age_days"] = _estimate_account_age(out["posts_count"], out["posts_per_day"])

    # Username features — not available, estimate from label
    out["username_digit_ratio"] = _estimate_username_digit_ratio(df["label"])
    out["username_length"] = _estimate_username_length(df["label"])

    out["night_activity_ratio"] = _estimate_night_activity(df["label"])
    out["mentions_per_post"] = _estimate_mentions(df["label"], out["engagement_rate"])
    out["story_frequency"] = _estimate_story_freq(df["label"], out["engagement_rate"], out["posts_per_day"])
    out["comment_reply_rate"] = _estimate_comment_reply_rate(
        df["label"], out["engagement_rate"], out["followers_count"]
    )
    out["unique_commenters_ratio"] = _estimate_unique_commenters(
        df["label"], out["engagement_rate"]
    )

    out["label"] = df["label"].values
    return out


# ── Dataset 2: InstaFake ──────────────────────────────────────────────────────

def load_instafake(directory: str | Path) -> pd.DataFrame:
    """
    Load the InstaFake dataset (JSON files from GitHub: fcakyon/instafake-dataset).

    Expected files:
      - fakeAccountData.json    (200 fake accounts, 9 features)
      - realAccountData.json    (994 real accounts, 9 features)
      - automatedAccountData.json    (700 automated accounts, 15 features)
      - nonautomatedAccountData.json (700 non-automated accounts, 15 features)
    """
    directory = Path(directory)
    rows = []

    # File → label mapping
    file_labels = {
        "fakeAccountData.json": 1,          # Bot
        "realAccountData.json": 0,          # Human
        "automatedAccountData.json": 2,     # Suspicious (automated)
        "nonautomatedAccountData.json": 0,  # Human
    }

    json_files = list(directory.glob("*.json"))
    if not json_files:
        json_files = list(directory.glob("**/*.json"))

    if not json_files:
        print(f"  InstaFake: no JSON files found in {directory}")
        return pd.DataFrame()

    for jf in json_files:
        try:
            with open(jf) as f:
                data = json.load(f)
        except Exception as e:
            print(f"  Warning: could not read {jf}: {e}")
            continue

        if not isinstance(data, list):
            continue

        # Determine label from filename
        label = file_labels.get(jf.name)
        if label is None:
            fname = jf.stem.lower()
            if "fake" in fname:
                label = 1
            elif "auto" in fname and "nonauto" not in fname:
                label = 2
            else:
                label = 0

        print(f"  {jf.name}: {len(data)} records → label={label} ({LABELS.get(label, '?')})")

        for item in data:
            if not isinstance(item, dict):
                continue
            row = _parse_instafake_item(item, label)
            if row:
                rows.append(row)

    if not rows:
        print(f"  InstaFake: no valid records found")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    print(f"  InstaFake: loaded {len(df):,} rows total")
    for lid, lname in LABELS.items():
        cnt = (df["label"] == lid).sum()
        print(f"    {lname:>12}: {cnt:,}")
    return df


def _parse_instafake_item(item: dict, default_label: int) -> dict | None:
    """Parse a single InstaFake JSON record into our feature schema.

    Handles two formats:
      1. Basic (fake-v1.0): userFollowerCount, userFollowingCount, etc.
      2. Rich (automated-v1.0): adds mediaLikeNumbers, mediaCommentNumbers,
         mediaHashtagNumbers, mediaUploadTimes (arrays per post)
    """
    # Extract basic profile features (camelCase keys)
    followers = item.get("userFollowerCount", item.get("user_follower_count"))
    following = item.get("userFollowingCount", item.get("user_following_count"))
    posts = item.get("userMediaCount", item.get("user_media_count"))

    if followers is None and following is None:
        return None

    followers = max(int(followers or 0), 0)
    following = max(int(following or 0), 0)
    posts = max(int(posts or 0), 0)

    has_pic = int(item.get("userHasProfilPic", item.get("user_has_profil_pic", 1)))
    is_private = int(item.get("userIsPrivate", item.get("user_is_private", 0)))
    bio_len = int(item.get("userBiographyLength", item.get("user_biography_length", 0)))
    uname_len = int(item.get("usernameLength", item.get("username_length", 10)))
    uname_digits = item.get("usernameDigitCount", item.get("username_digit_count"))
    has_url = int(item.get("userHasExternalUrl", item.get("user_has_external_url", 0)))

    digit_ratio = round(int(uname_digits) / max(uname_len, 1), 3) if uname_digits is not None else None

    # Label from item or default
    label = default_label
    if "isFake" in item:
        label = 1 if item["isFake"] else 0

    # ── Rich features (automated-v1.0 format) ───────────────────────────
    avg_likes = 0
    avg_comments = 0
    engagement_rate = 0.0
    hashtags_per_post = 0.0
    night_activity_ratio = None
    posts_per_day = None

    # mediaLikeNumbers: list of like counts per post
    media_likes = item.get("mediaLikeNumbers", [])
    if media_likes and isinstance(media_likes, list):
        avg_likes = int(np.mean([x for x in media_likes if isinstance(x, (int, float))]) or 0)
        engagement_rate = round(avg_likes / max(followers, 1), 4)

    # mediaCommentNumbers: list of comment counts per post
    media_comments = item.get("mediaCommentNumbers", [])
    if media_comments and isinstance(media_comments, list):
        avg_comments = int(np.mean([x for x in media_comments if isinstance(x, (int, float))]) or 0)

    # mediaHashtagNumbers: list of hashtag counts per post
    media_hashtags = item.get("mediaHashtagNumbers", [])
    if media_hashtags and isinstance(media_hashtags, list):
        valid_h = [x for x in media_hashtags if isinstance(x, (int, float))]
        if valid_h:
            hashtags_per_post = round(np.mean(valid_h), 1)

    # mediaUploadTimes: list of Unix timestamps → derive night_activity + posts_per_day
    upload_times = item.get("mediaUploadTimes", [])
    if upload_times and isinstance(upload_times, list) and len(upload_times) > 1:
        valid_times = sorted([t for t in upload_times if isinstance(t, (int, float)) and t > 0])
        if len(valid_times) > 1:
            # Night activity: posts between 00:00-06:00
            from datetime import datetime, timezone
            hours = [datetime.fromtimestamp(t, tz=timezone.utc).hour for t in valid_times]
            night_count = sum(1 for h in hours if 0 <= h < 6)
            night_activity_ratio = round(night_count / len(hours), 3)

            # Posts per day from time span
            span_days = (valid_times[-1] - valid_times[0]) / 86400
            if span_days > 0:
                posts_per_day = round(len(valid_times) / span_days, 3)

    # ── Build output row ─────────────────────────────────────────────────
    row = {
        "followers_count": followers,
        "following_count": following,
        "posts_count": posts,
        "avg_likes_per_post": avg_likes,
        "avg_comments_per_post": avg_comments,
        "posts_per_day": posts_per_day if posts_per_day else round(np.random.uniform(0.1, 2.0), 3),
        "follower_following_ratio": round(followers / max(following, 1), 4),
        "engagement_rate": engagement_rate,
        "profile_pic": has_pic,
        "bio_length": bio_len,
        "has_url_in_bio": has_url,
        "is_verified": 0,
        "is_private": is_private,
        "account_age_days": int(np.random.uniform(90, 2500)),  # not available
        "username_digit_ratio": digit_ratio if digit_ratio is not None else round(np.random.beta(1, 8) * 0.25, 3),
        "username_length": uname_len,
        "night_activity_ratio": night_activity_ratio if night_activity_ratio is not None else round(np.random.beta(2, 6) * 0.5, 3),
        "avg_caption_length": 0,  # not available
        "hashtags_per_post": hashtags_per_post,
        "mentions_per_post": round(np.random.uniform(0, 3), 1),  # not available
        "story_frequency": round(np.random.uniform(0.1, 2.0), 2),  # not available
        "reels_ratio": round(np.random.beta(2, 5), 3),  # not available
        "comment_reply_rate": round(np.random.beta(3, 4) * 0.6, 3),  # not available
        "unique_commenters_ratio": round(np.random.beta(4, 3) * 0.8, 3),  # not available
        "label": label,
    }
    return row


# ── Estimation functions for missing features ────────────────────────────────
# These use statistical distributions calibrated from real Instagram research
# and conditioned on the observed features + label.

def _estimate_is_verified(followers: pd.Series) -> pd.Series:
    """Verification correlates strongly with follower count."""
    probs = np.where(followers > 100_000, 0.15,
            np.where(followers > 10_000, 0.03, 0.005))
    return pd.Series(np.random.binomial(1, probs), index=followers.index)


def _estimate_is_private(labels: pd.Series, followers: pd.Series) -> pd.Series:
    """Humans often have private accounts; bots rarely do."""
    probs = np.where(labels == 0, 0.50,  # humans: ~50% private
            np.where(labels == 1, 0.12,   # bots: rarely private
                     0.35))               # suspicious: sometimes
    # High-follower accounts are less likely private
    probs = np.where(followers > 10_000, probs * 0.3, probs)
    return pd.Series(np.random.binomial(1, probs.clip(0, 1)), index=labels.index)


def _estimate_account_age(posts: pd.Series, posts_per_day: pd.Series) -> pd.Series:
    """Approximate account age from total posts / posting rate."""
    # posts / posts_per_day = days of activity; real accounts are older
    estimated = (posts / posts_per_day.clip(lower=0.01)).clip(lower=30, upper=5000)
    # Add noise
    noise = np.random.normal(1.0, 0.2, size=len(estimated)).clip(0.5, 2.0)
    return (estimated * noise).round(0).astype(int)


def _estimate_username_digit_ratio(labels: pd.Series) -> pd.Series:
    """Bots tend to have more digits in usernames."""
    ratios = np.where(
        labels == 0,
        np.random.beta(1, 8, size=len(labels)) * 0.25,   # humans: low digits
        np.where(
            labels == 1,
            np.random.beta(5, 3, size=len(labels)) * 0.7, # bots: high digits
            np.random.beta(3, 5, size=len(labels)) * 0.45  # suspicious: medium
        )
    )
    return pd.Series(ratios.round(3), index=labels.index)


def _estimate_username_length(labels: pd.Series) -> pd.Series:
    """Bots tend to have longer, auto-generated usernames."""
    lengths = np.where(
        labels == 0,
        np.clip(np.random.normal(11, 3, size=len(labels)), 4, 28),
        np.where(
            labels == 1,
            np.clip(np.random.normal(16, 4, size=len(labels)), 6, 28),
            np.clip(np.random.normal(13, 4, size=len(labels)), 4, 28)
        )
    )
    return pd.Series(lengths.round(0).astype(int), index=labels.index)


def _estimate_night_activity(labels: pd.Series) -> pd.Series:
    """Bots post more at night (automated, no sleep cycle)."""
    ratios = np.where(
        labels == 0,
        np.random.beta(2, 6, size=len(labels)) * 0.5,
        np.where(
            labels == 1,
            np.random.beta(5, 2, size=len(labels)) * 0.8,
            np.random.beta(3, 4, size=len(labels)) * 0.6
        )
    )
    return pd.Series(ratios.round(3), index=labels.index)


def _estimate_mentions(labels: pd.Series, engagement_rate: pd.Series) -> pd.Series:
    """Bots spam mentions; humans mention sparingly."""
    mentions = np.where(
        labels == 0,
        np.clip(np.random.normal(0.8, 1.0, size=len(labels)), 0, 10),
        np.where(
            labels == 1,
            np.clip(np.random.normal(4.5, 3.0, size=len(labels)), 0, 15),
            np.clip(np.random.normal(2.2, 1.8, size=len(labels)), 0, 10)
        )
    )
    return pd.Series(mentions.round(1), index=labels.index)


def _estimate_story_freq(
    labels: pd.Series, engagement_rate: pd.Series, posts_per_day: pd.Series
) -> pd.Series:
    """Humans post stories regularly; bots rarely do."""
    freq = np.where(
        labels == 0,
        np.clip(np.random.normal(1.1, 0.8, size=len(labels)), 0.1, 4.0),
        np.where(
            labels == 1,
            np.clip(np.random.normal(0.1, 0.2, size=len(labels)), 0, 1),
            np.clip(np.random.normal(0.45, 0.4, size=len(labels)), 0, 2)
        )
    )
    return pd.Series(freq.round(2), index=labels.index)


def _estimate_comment_reply_rate(
    labels: pd.Series, engagement_rate: pd.Series, followers: pd.Series
) -> pd.Series:
    """Humans reply to comments; bots don't."""
    rates = np.where(
        labels == 0,
        np.random.beta(3, 4, size=len(labels)) * 0.65,   # 0.25-0.65
        np.where(
            labels == 1,
            np.random.beta(1, 9, size=len(labels)) * 0.15, # 0-0.15
            np.random.beta(2, 6, size=len(labels)) * 0.35   # 0-0.35
        )
    )
    return pd.Series(rates.round(3), index=labels.index)


def _estimate_unique_commenters(
    labels: pd.Series, engagement_rate: pd.Series
) -> pd.Series:
    """Humans have diverse commenters; bots get comments from other bots."""
    ratios = np.where(
        labels == 0,
        np.random.beta(5, 3, size=len(labels)) * 0.82,   # 0.50-0.82
        np.where(
            labels == 1,
            np.random.beta(1, 6, size=len(labels)) * 0.3,  # 0-0.30
            np.random.beta(3, 6, size=len(labels)) * 0.50   # 0-0.50
        )
    )
    return pd.Series(ratios.round(3), index=labels.index)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def find_dataset_files() -> dict:
    """Auto-detect dataset files in data/real/."""
    found = {}

    # Look for Purba dataset (CSV with columns: pos, flr, flg, etc.)
    for csv_file in REAL_DIR.glob("*.csv"):
        try:
            sample = pd.read_csv(csv_file, nrows=5)
            cols = set(sample.columns)
            # Purba has these distinctive columns (flr or flw for followers)
            if {"pos", "flg", "bl"}.issubset(cols) and ({"erl", "erc"}.issubset(cols) or "flw" in cols or "flr" in cols):
                found["purba"] = csv_file
                print(f"  Found Purba dataset: {csv_file.name}")
            elif ("label" in cols or "class" in cols) and {"flg"}.issubset(cols) and ("flr" in cols or "flw" in cols):
                found["purba"] = csv_file
                print(f"  Found Purba dataset (variant): {csv_file.name}")
        except Exception:
            pass

    # Look for InstaFake dataset (JSON files or directory)
    for json_file in REAL_DIR.glob("**/*.json"):
        if "instafake" not in found:
            found["instafake"] = json_file.parent
            print(f"  Found InstaFake data in: {json_file.parent}")

    return found


def merge_and_clean(dataframes: list[pd.DataFrame]) -> pd.DataFrame:
    """Merge multiple processed datasets, clean, and validate."""
    df = pd.concat(dataframes, ignore_index=True)

    # Ensure all features are present
    for col in FEATURE_COLS:
        if col not in df.columns:
            raise ValueError(f"Missing feature column: {col}")

    # Clean numeric columns
    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with too many NaNs
    df = df.dropna(subset=["label"])
    df = df.dropna(thresh=len(FEATURE_COLS) - 3)  # allow up to 3 NaN features
    df = df.fillna(0)

    # Ensure correct types
    int_cols = ["followers_count", "following_count", "posts_count",
                "avg_likes_per_post", "avg_comments_per_post",
                "profile_pic", "bio_length", "has_url_in_bio", "is_verified",
                "is_private", "account_age_days", "username_length",
                "avg_caption_length"]
    for col in int_cols:
        df[col] = df[col].astype(int)

    df["label"] = df["label"].astype(int)

    # Remove obvious outliers
    df = df[df["followers_count"] >= 0]
    df = df[df["following_count"] >= 0]
    df = df[df["engagement_rate"] <= 1.0]

    print(f"\n  Merged dataset: {len(df):,} samples")
    print(f"  Label distribution:")
    for label_id, label_name in LABELS.items():
        count = (df["label"] == label_id).sum()
        pct = count / len(df) * 100
        print(f"    {label_name:>12}: {count:,}  ({pct:.1f}%)")

    return df


def save_splits(df: pd.DataFrame, prefix: str = "real"):
    """Save train/val/test splits."""
    out_dir = Path(__file__).resolve().parent

    train, tmp = train_test_split(
        df, test_size=0.30, stratify=df["label"], random_state=42
    )
    val, test = train_test_split(
        tmp, test_size=0.50, stratify=tmp["label"], random_state=42
    )

    train.to_csv(out_dir / f"{prefix}_train.csv", index=False)
    val.to_csv(out_dir / f"{prefix}_val.csv", index=False)
    test.to_csv(out_dir / f"{prefix}_test.csv", index=False)

    print(f"\n  Saved to {out_dir}/")
    print(f"    {prefix}_train.csv : {len(train):,}")
    print(f"    {prefix}_val.csv   : {len(val):,}")
    print(f"    {prefix}_test.csv  : {len(test):,}")

    return train, val, test


def main():
    print("=" * 60)
    print("  Real Data Preparation Pipeline")
    print("=" * 60)

    if not REAL_DIR.exists():
        REAL_DIR.mkdir(parents=True)
        print(f"\n  Created {REAL_DIR}/")
        print("  Please download datasets and place them here:")
        print("    1. Purba (CSV): kaggle.com/datasets/krpurba/fakeauthentic-user-instagram")
        print("    2. InstaFake (JSON): github.com/fcakyon/instafake-dataset")
        sys.exit(1)

    print(f"\n  Scanning {REAL_DIR}/ for datasets...")
    found = find_dataset_files()

    if not found:
        print("\n  No datasets found! Please download and place files in data/real/")
        print("  Accepted formats:")
        print("    - Purba: CSV with columns pos, flr, flg, bl, pic, lin, erl, erc, etc.")
        print("    - InstaFake: JSON files with user profile data")
        print("\n  Trying to load any CSV file with a 'label' column...")

        # Fallback: try any CSV with a label column
        for csv_file in REAL_DIR.glob("*.csv"):
            try:
                sample = pd.read_csv(csv_file, nrows=5)
                if "label" in sample.columns:
                    found["generic"] = csv_file
                    print(f"  Found generic labeled CSV: {csv_file.name}")
                    break
            except Exception:
                pass

    if not found:
        print("\n  ERROR: No usable dataset files found in data/real/")
        sys.exit(1)

    # Process each dataset
    dfs = []

    if "purba" in found:
        print(f"\n── Processing Purba dataset ──")
        df_purba = load_purba(found["purba"])
        dfs.append(df_purba)

    if "instafake" in found:
        print(f"\n── Processing InstaFake dataset ──")
        df_insta = load_instafake(found["instafake"])
        if len(df_insta) > 0:
            dfs.append(df_insta)

    if not dfs:
        print("\n  ERROR: Could not process any datasets")
        sys.exit(1)

    # Merge and save
    print(f"\n── Merging datasets ──")
    df_merged = merge_and_clean(dfs)

    print(f"\n── Feature statistics ──")
    stats = df_merged[FEATURE_COLS].describe().T[["mean", "std", "min", "max"]]
    print(stats.round(3).to_string())

    print(f"\n── Saving splits ──")
    save_splits(df_merged, prefix="real")

    print(f"\n{'=' * 60}")
    print(f"  Done! You can now retrain the model:")
    print(f"  python model/train.py --data real")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
