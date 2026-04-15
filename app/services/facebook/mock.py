"""
Mock Facebook provider — returns realistic fake data for testing.
Use this for local development (FACEBOOK_PROVIDER=mock).
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
        "full_name": "Michael Chen",
        "biography": "Software engineer at BigTech. Dog dad. Bay Area.",
        "followers_count": 1_200,
        "following_count": 850,
        "posts_count": 340,
        "posts": [
            {"likes": 85, "comments": 12, "text": "Beautiful sunset from the office rooftop today!"},
            {"likes": 120, "comments": 25, "text": "Happy birthday to this amazing person! So grateful for you."},
            {"likes": 45, "comments": 6, "text": "Weekend hike at Muir Woods. Nature therapy is real."},
        ],
    },
    "bot_test": {
        "full_name": "Free iPhone Giveaway",
        "biography": "",
        "followers_count": 120,
        "following_count": 4_900,
        "posts_count": 15,
        "posts": [
            {"likes": 0, "comments": 0, "text": "Click here!"},
            {"likes": 1, "comments": 0, "text": "FREE"},
        ],
    },
    "suspicious_test": {
        "full_name": "Digital Marketing Pro",
        "biography": "10x your followers! DM for packages. Results guaranteed.",
        "followers_count": 15_000,
        "following_count": 8_000,
        "posts_count": 1_500,
        "posts": [
            {"likes": 30, "comments": 3, "text": "5 secrets to grow your page FAST! #marketing #growth"},
            {"likes": 20, "comments": 1, "text": "Another client hitting 10K! DM me for details."},
        ],
    },
}


class FacebookMockProvider(SocialMediaProvider):
    """Returns deterministic fake Facebook data — no network calls."""

    async def get_profile(self, username: str) -> RawSocialProfile:
        if username == "nonexistent_user_404":
            raise InstagramUserNotFoundError(f"@{username} does not exist on Facebook.")

        preset = _PRESETS.get(username)
        if preset:
            return self._build_from_preset(username, preset)
        return self._build_random(username)

    def _build_from_preset(self, username: str, p: dict) -> RawSocialProfile:
        now = datetime.now(_UTC)
        posts = []
        for i, post in enumerate(p["posts"]):
            posts.append(PostData(
                likes_count=post["likes"],
                comments_count=post["comments"],
                caption=post["text"],
                media_type="Image",
                timestamp=now - timedelta(days=i * 7 + 1),
            ))
        return RawSocialProfile(
            platform=Platform.FACEBOOK,
            username=username,
            full_name=p["full_name"],
            biography=p["biography"],
            profile_pic_url=f"https://mock-cdn.botscan.dev/facebook/{username}.jpg",
            followers_count=p["followers_count"],
            following_count=p["following_count"],
            posts_count=p["posts_count"],
            is_private=False,
            recent_posts=posts,
        )

    def _build_random(self, username: str) -> RawSocialProfile:
        rng = random.Random(f"facebook_{username}")
        now = datetime.now(_UTC)
        followers = int(rng.lognormvariate(6.0, 1.5))
        following = int(rng.lognormvariate(5.5, 1.0))
        posts_count = int(rng.lognormvariate(4.5, 1.5))
        posts = []
        for i in range(min(10, posts_count)):
            days_back = rng.randint(i * 3 + 1, max(i * 10 + 14, 30))
            likes = int(followers * rng.betavariate(2, 6) * 0.08)
            comments = max(0, int(likes * rng.betavariate(1, 5) * 0.12))
            posts.append(PostData(
                likes_count=likes, comments_count=comments,
                caption=" ".join(rng.choice(["great", "fun", "family", "love", "weekend"]) for _ in range(rng.randint(3, 15))),
                media_type="Image",
                timestamp=now - timedelta(days=days_back),
            ))
        return RawSocialProfile(
            platform=Platform.FACEBOOK,
            username=username,
            full_name=username.replace(".", " ").title(),
            biography=" ".join(rng.choice(["family", "sports", "cooking", "travel"]) for _ in range(rng.randint(0, 8))),
            profile_pic_url=f"https://mock-cdn.botscan.dev/facebook/{username}.jpg",
            followers_count=max(0, followers),
            following_count=max(0, following),
            posts_count=max(0, posts_count),
            is_private=rng.random() < 0.55,
            recent_posts=posts,
        )
