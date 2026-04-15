"""
Mock Instagram provider — returns realistic fake data without any API key.
Use this for local development and testing (INSTAGRAM_PROVIDER=mock).

Preset accounts:
  "human_test"      → clearly human
  "bot_test"        → clearly bot
  "suspicious_test" → suspicious
  anything else     → semi-random realistic profile
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

from app.core.exceptions import InstagramUserNotFoundError
from app.schemas.social import Platform, PostData, RawSocialProfile
from app.services.instagram.base import InstagramProvider

_UTC = timezone.utc

# ── Preset fixtures ────────────────────────────────────────────────────────────

_PRESETS: dict[str, dict] = {
    "human_test": {
        "full_name": "Emma Johnson",
        "biography": "Travel photographer based in Paris. Coffee addict. She/her.",
        "followers_count": 4_200,
        "following_count": 612,
        "posts_count": 187,
        "is_verified": False,
        "is_private": False,
        "is_business": False,
        "external_url": "https://emmajohnson.photography",
        "days_ago_oldest_post": 730,
        "posts": [
            {"likes": 340, "comments": 22, "caption": "Golden hour in Montmartre #paris #travel #photography", "is_video": False},
            {"likes": 280, "comments": 18, "caption": "Espresso in the morning ☕ @cafedelamere", "is_video": False},
            {"likes": 510, "comments": 41, "caption": "Behind the lens — new project coming soon! #photographer #artofvisuals", "is_video": True},
            {"likes": 190, "comments": 9,  "caption": "Weekend vibes 🌿", "is_video": False},
            {"likes": 420, "comments": 35, "caption": "When the light hits just right ✨ #goldenlight", "is_video": False},
            {"likes": 310, "comments": 14, "caption": "Rainy day reads 📚 #bookstagram", "is_video": False},
        ],
    },
    "bot_test": {
        "full_name": "user82934712",
        "biography": "Follow for follow! DM for promo 🔥",
        "followers_count": 342,
        "following_count": 4_891,
        "posts_count": 23,
        "is_verified": False,
        "is_private": False,
        "is_business": False,
        "external_url": "https://bit.ly/3xPromo",
        "days_ago_oldest_post": 45,
        "posts": [
            {"likes": 2,  "comments": 0, "caption": "#follow #followme #followback #like #likeback #f4f #l4l #followforfollow", "is_video": False},
            {"likes": 1,  "comments": 0, "caption": "#instagood #instadaily #follow4follow #like4like #photooftheday", "is_video": False},
            {"likes": 3,  "comments": 1, "caption": "DM me for shoutout 🔥🔥 #promo #promotion", "is_video": False},
            {"likes": 0,  "comments": 0, "caption": "#f4f #l4l", "is_video": False},
            {"likes": 2,  "comments": 0, "caption": "#follow #like #trending", "is_video": False},
            {"likes": 1,  "comments": 0, "caption": "Best offer!! DM now 💰 #money #cash", "is_video": False},
        ],
    },
    "suspicious_test": {
        "full_name": "Growth Hacks Daily",
        "biography": "Social media tips & tricks. 10k followers in 30 days guaranteed!",
        "followers_count": 18_400,
        "following_count": 9_200,
        "posts_count": 312,
        "is_verified": False,
        "is_private": False,
        "is_business": True,
        "external_url": "https://growthacks.link/course",
        "days_ago_oldest_post": 200,
        "posts": [
            {"likes": 85,  "comments": 4,  "caption": "10 tips to go viral 🚀 #socialmedia #growth #viral #marketing", "is_video": False},
            {"likes": 120, "comments": 8,  "caption": "How I gained 5000 followers in a week (thread) 👇 #instagram #tips", "is_video": True},
            {"likes": 60,  "comments": 2,  "caption": "#growthhacking #instagramtips #followers #likes #viral #trending", "is_video": False},
            {"likes": 95,  "comments": 5,  "caption": "Algorithm secrets they don't want you to know 🤫", "is_video": False},
            {"likes": 70,  "comments": 3,  "caption": "My morning routine for productivity 💪 #motivation #hustle", "is_video": False},
        ],
    },
}


class MockProvider(InstagramProvider):
    """Returns deterministic fake data — no network calls."""

    async def get_profile(self, username: str) -> RawSocialProfile:
        if username == "nonexistent_user_404":
            raise InstagramUserNotFoundError(f"@{username} does not exist.")

        preset = _PRESETS.get(username)
        if preset:
            return self._build_from_preset(username, preset)

        # Unknown username → generate a plausible random profile
        return self._build_random(username)

    # ── Private helpers ────────────────────────────────────────────────────

    def _build_from_preset(self, username: str, p: dict) -> RawSocialProfile:
        now = datetime.now(_UTC)
        oldest_days = p["days_ago_oldest_post"]
        posts: list[PostData] = []
        for i, pd in enumerate(p["posts"]):
            days_back = random.randint(i * 5, oldest_days)
            posts.append(PostData(
                likes_count=pd["likes"],
                comments_count=pd["comments"],
                caption=pd["caption"],
                is_video=pd["is_video"],
                media_type="Video" if pd["is_video"] else "Image",
                timestamp=now - timedelta(days=days_back),
            ))
        return RawSocialProfile(
            platform=Platform.INSTAGRAM,
            username=username,
            full_name=p["full_name"],
            biography=p["biography"],
            profile_pic_url=f"https://mock-cdn.botscan.dev/{username}.jpg",
            followers_count=p["followers_count"],
            following_count=p["following_count"],
            posts_count=p["posts_count"],
            is_verified=p["is_verified"],
            is_private=p["is_private"],
            is_business=p["is_business"],
            external_url=p["external_url"],
            recent_posts=posts,
        )

    def _build_random(self, username: str) -> RawSocialProfile:
        rng = random.Random(username)  # deterministic for the same username
        now = datetime.now(_UTC)

        followers = int(rng.lognormvariate(7.0, 1.5))
        following = int(rng.lognormvariate(5.8, 0.9))
        posts_count = int(rng.lognormvariate(4.0, 1.2))
        bio_words = rng.randint(0, 20)
        biography = " ".join(
            rng.choice(["travel", "food", "fitness", "art", "tech", "music", "life", "love", "creator"])
            for _ in range(bio_words)
        )

        posts: list[PostData] = []
        for i in range(min(12, posts_count)):
            days_back = rng.randint(i * 3 + 1, max(i * 10 + 30, 60))
            likes = int(followers * rng.betavariate(2, 6) * 0.1)
            comments = max(0, int(likes * rng.betavariate(1, 8) * 0.1))
            hashtag_count = rng.randint(0, 15)
            caption = " ".join(
                [rng.choice(["great", "amazing", "love", "wow", "beautiful"]) for _ in range(rng.randint(3, 15))]
                + [f"#tag{j}" for j in range(hashtag_count)]
            )
            is_video = rng.random() < 0.3
            posts.append(PostData(
                likes_count=likes,
                comments_count=comments,
                caption=caption,
                is_video=is_video,
                media_type="Video" if is_video else "Image",
                timestamp=now - timedelta(days=days_back),
            ))

        return RawSocialProfile(
            platform=Platform.INSTAGRAM,
            username=username,
            full_name=username.replace("_", " ").title(),
            biography=biography,
            profile_pic_url=f"https://mock-cdn.botscan.dev/{username}.jpg",
            followers_count=max(0, followers),
            following_count=max(0, following),
            posts_count=max(0, posts_count),
            is_verified=rng.random() < 0.03,
            is_private=rng.random() < 0.40,
            is_business=rng.random() < 0.15,
            external_url="https://example.com" if rng.random() < 0.3 else None,
            recent_posts=posts,
        )
