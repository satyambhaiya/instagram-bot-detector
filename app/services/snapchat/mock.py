"""
Mock Snapchat provider — returns realistic fake data for testing.
Use this for local development (SNAPCHAT_PROVIDER=mock).

Note: Snapchat has very limited public profile data compared to other platforms.
The mock reflects this reality — fewer features are directly observable.
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
        "full_name": "Alex Rivera",
        "biography": "snap me 👻",
        "followers_count": 450,
        "following_count": 380,
        "posts_count": 1_200,  # snap score approximation
    },
    "bot_test": {
        "full_name": "add4add",
        "biography": "add me back! premium content 🔥 DM",
        "followers_count": 5_000,
        "following_count": 4_900,
        "posts_count": 50,
    },
    "suspicious_test": {
        "full_name": "SnapDeals Official",
        "biography": "Exclusive deals & promos! Add us! 💰",
        "followers_count": 12_000,
        "following_count": 200,
        "posts_count": 8_000,
    },
}


class SnapchatMockProvider(SocialMediaProvider):
    """Returns deterministic fake Snapchat data — no network calls."""

    async def get_profile(self, username: str) -> RawSocialProfile:
        if username == "nonexistent_user_404":
            raise InstagramUserNotFoundError(f"@{username} does not exist on Snapchat.")

        preset = _PRESETS.get(username)
        if preset:
            return self._build_from_preset(username, preset)
        return self._build_random(username)

    def _build_from_preset(self, username: str, p: dict) -> RawSocialProfile:
        return RawSocialProfile(
            platform=Platform.SNAPCHAT,
            username=username,
            full_name=p["full_name"],
            biography=p.get("biography", ""),
            profile_pic_url=f"https://mock-cdn.botscan.dev/snapchat/{username}.jpg",
            followers_count=p["followers_count"],
            following_count=p["following_count"],
            posts_count=p["posts_count"],
            is_private=False,
            recent_posts=[],  # Snapchat stories are ephemeral
        )

    def _build_random(self, username: str) -> RawSocialProfile:
        rng = random.Random(f"snapchat_{username}")
        followers = int(rng.lognormvariate(5.5, 1.5))
        following = int(rng.lognormvariate(5.3, 1.0))
        snap_score = int(rng.lognormvariate(7.0, 2.0))
        return RawSocialProfile(
            platform=Platform.SNAPCHAT,
            username=username,
            full_name=username.replace("_", " ").title(),
            biography="" if rng.random() < 0.6 else "add me!",
            profile_pic_url=f"https://mock-cdn.botscan.dev/snapchat/{username}.jpg",
            followers_count=max(0, followers),
            following_count=max(0, following),
            posts_count=max(0, snap_score),
            is_private=rng.random() < 0.3,
            recent_posts=[],
        )
