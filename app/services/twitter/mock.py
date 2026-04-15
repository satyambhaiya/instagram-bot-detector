"""
Mock Twitter/X provider — returns realistic fake data for testing.
Use this for local development (TWITTER_PROVIDER=mock).
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

from app.core.exceptions import InstagramUserNotFoundError
from app.schemas.social import Platform, PostData, RawSocialProfile
from app.services.social_base import SocialMediaProvider

_UTC = timezone.utc

_PRESETS: dict[str, dict] = {
    "human_test": {
        "full_name": "Sarah Mitchell",
        "biography": "Tech writer | Coffee enthusiast | Views are my own",
        "followers_count": 2_800,
        "following_count": 890,
        "posts_count": 4_200,
        "is_verified": False,
        "tweets": [
            {"likes": 45, "comments": 8, "text": "Just published my latest article on AI trends for 2025. Link in bio! #tech #AI"},
            {"likes": 23, "comments": 3, "text": "Morning coffee and code. The perfect combo."},
            {"likes": 120, "comments": 22, "text": "@elonmusk thoughts on the new API changes? This affects a lot of developers."},
            {"likes": 15, "comments": 1, "text": "Great talk at #TechConf2024 today. Learned so much about distributed systems."},
        ],
    },
    "bot_test": {
        "full_name": "CryptoGains247",
        "biography": "🚀 FREE Bitcoin! DM for details 💰 #crypto #forex #trading",
        "followers_count": 156,
        "following_count": 4_980,
        "posts_count": 12_000,
        "is_verified": False,
        "tweets": [
            {"likes": 0, "comments": 0, "text": "EARN $500/DAY! Click here: bit.ly/scam123 #bitcoin #crypto #money #rich"},
            {"likes": 1, "comments": 0, "text": "#follow #followback #f4f #crypto #bitcoin #ethereum #trading #money #rich #invest"},
            {"likes": 0, "comments": 0, "text": "DM me for FREE crypto signals! 💰💰💰 #forex #trading"},
        ],
    },
    "suspicious_test": {
        "full_name": "News Bot Daily",
        "biography": "Automated news aggregator. Retweets trending topics 24/7.",
        "followers_count": 8_500,
        "following_count": 5_200,
        "posts_count": 25_000,
        "is_verified": False,
        "tweets": [
            {"likes": 12, "comments": 1, "text": "BREAKING: Latest developments in tech sector. #news #breaking"},
            {"likes": 8, "comments": 0, "text": "Top 10 trending topics today #trending #viral"},
            {"likes": 5, "comments": 0, "text": "Market update: stocks rise. #stocks #market #finance"},
        ],
    },
}


class TwitterMockProvider(SocialMediaProvider):
    """Returns deterministic fake Twitter data — no network calls."""

    async def get_profile(self, username: str) -> RawSocialProfile:
        if username == "nonexistent_user_404":
            raise InstagramUserNotFoundError(f"@{username} does not exist on Twitter.")

        preset = _PRESETS.get(username)
        if preset:
            return self._build_from_preset(username, preset)
        return self._build_random(username)

    def _build_from_preset(self, username: str, p: dict) -> RawSocialProfile:
        now = datetime.now(_UTC)
        posts = []
        for i, t in enumerate(p["tweets"]):
            posts.append(PostData(
                likes_count=t["likes"],
                comments_count=t["comments"],
                caption=t["text"],
                is_video=False,
                media_type="Tweet",
                timestamp=now - timedelta(days=i * 2 + 1),
            ))
        return RawSocialProfile(
            platform=Platform.TWITTER,
            username=username,
            full_name=p["full_name"],
            biography=p["biography"],
            profile_pic_url=f"https://mock-cdn.botscan.dev/twitter/{username}.jpg",
            followers_count=p["followers_count"],
            following_count=p["following_count"],
            posts_count=p["posts_count"],
            is_verified=p["is_verified"],
            is_private=False,
            external_url=None,
            recent_posts=posts,
        )

    def _build_random(self, username: str) -> RawSocialProfile:
        rng = random.Random(f"twitter_{username}")
        now = datetime.now(_UTC)
        followers = int(rng.lognormvariate(6.5, 1.8))
        following = int(rng.lognormvariate(5.5, 1.0))
        tweets_count = int(rng.lognormvariate(7.0, 2.0))
        posts = []
        for i in range(min(12, tweets_count)):
            days_back = rng.randint(i + 1, max(i * 5 + 10, 30))
            likes = int(followers * rng.betavariate(2, 8) * 0.05)
            comments = max(0, int(likes * rng.betavariate(1, 6) * 0.15))
            htags = rng.randint(0, 8)
            text = " ".join(
                [rng.choice(["breaking", "thread", "opinion", "update", "news"]) for _ in range(rng.randint(5, 25))]
                + [f"#tag{j}" for j in range(htags)]
            )
            posts.append(PostData(
                likes_count=likes, comments_count=comments,
                caption=text, media_type="Tweet",
                timestamp=now - timedelta(days=days_back),
            ))
        return RawSocialProfile(
            platform=Platform.TWITTER,
            username=username,
            full_name=username.replace("_", " ").title(),
            biography=" ".join(rng.choice(["tech", "sports", "news", "life"]) for _ in range(rng.randint(0, 10))),
            profile_pic_url=f"https://mock-cdn.botscan.dev/twitter/{username}.jpg",
            followers_count=max(0, followers),
            following_count=max(0, following),
            posts_count=max(0, tweets_count),
            is_verified=rng.random() < 0.02,
            is_private=rng.random() < 0.15,
            recent_posts=posts,
        )
