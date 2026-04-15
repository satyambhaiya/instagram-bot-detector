"""
Apify Instagram Profile Scraper provider.

Requires: APIFY_API_TOKEN in .env
Actor:    apify/instagram-profile-scraper

Docs: https://apify.com/apify/instagram-profile-scraper
"""

from __future__ import annotations

from datetime import datetime, timezone

import httpx

from app.core.exceptions import (
    InstagramProviderError,
    InstagramUserNotFoundError,
    RateLimitError,
)
from app.core.logging import get_logger
from app.schemas.instagram import PostData, RawInstagramProfile
from app.services.instagram.base import InstagramProvider

logger = get_logger(__name__)

_APIFY_RUN_URL = (
    "https://api.apify.com/v2/acts/{actor_id}/run-sync-get-dataset-items"
)
_TIMEOUT = 60.0   # Apify runs can take ~20-30s; give generous timeout


class ApifyProvider(InstagramProvider):
    """Calls the Apify Instagram Profile Scraper synchronously."""

    def __init__(self, api_token: str, actor_id: str = "apify~instagram-profile-scraper"):
        if not api_token:
            raise ValueError("APIFY_API_TOKEN is required for the Apify provider.")
        self._token = api_token
        self._url = _APIFY_RUN_URL.format(actor_id=actor_id)

    async def get_profile(self, username: str) -> RawInstagramProfile:
        logger.info("Apify: fetching profile for @%s", username)
        params = {"token": self._token}
        payload = {"usernames": [username]}

        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            try:
                resp = await client.post(self._url, params=params, json=payload)
            except httpx.TimeoutException:
                raise InstagramProviderError(
                    "Apify request timed out. The actor may be overloaded."
                )
            except httpx.RequestError as exc:
                raise InstagramProviderError(f"Network error reaching Apify: {exc}")

        if resp.status_code == 429:
            raise RateLimitError("Apify rate limit reached.")
        if resp.status_code in (401, 403):
            raise InstagramProviderError("Invalid Apify API token.")
        if resp.status_code not in (200, 201):
            raise InstagramProviderError(
                f"Apify returned HTTP {resp.status_code}: {resp.text[:200]}"
            )

        items: list[dict] = resp.json()
        if not items:
            raise InstagramUserNotFoundError(f"@{username} was not found on Instagram.")

        return self._parse(items[0])

    # ── Parsing ───────────────────────────────────────────────────────────────

    def _parse(self, data: dict) -> RawInstagramProfile:
        posts: list[PostData] = []
        for p in data.get("latestPosts", []):
            ts = None
            raw_ts = p.get("timestamp")
            if raw_ts:
                try:
                    ts = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
                except ValueError:
                    pass
            posts.append(PostData(
                likes_count=p.get("likesCount", 0),
                comments_count=p.get("commentsCount", 0),
                caption=p.get("caption") or "",
                is_video=p.get("isVideo", False),
                media_type="Video" if p.get("isVideo") else p.get("type", "Image"),
                timestamp=ts,
            ))

        return RawInstagramProfile(
            username=data.get("username", ""),
            full_name=data.get("fullName", ""),
            biography=data.get("biography", ""),
            profile_pic_url=data.get("profilePicUrlHD") or data.get("profilePicUrl"),
            followers_count=data.get("followersCount", 0),
            following_count=data.get("followsCount", 0),
            posts_count=data.get("postsCount", 0),
            is_verified=data.get("verified", False),
            is_private=data.get("private", False),
            is_business=data.get("businessAccount", False),
            external_url=data.get("externalUrl") or None,
            recent_posts=posts,
        )
