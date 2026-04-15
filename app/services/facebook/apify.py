"""
Apify Facebook provider — fetches profile + recent posts via Apify actor.

Actor:  apify/facebook-posts-scraper
Docs:   https://apify.com/apify/facebook-posts-scraper

Returns posts with text, reactions (like/love/wow/etc.), comments, shares,
timestamps, and page metadata (name, profile pic, Facebook ID).

Usage (.env):
    FACEBOOK_PROVIDER=apify
    APIFY_API_TOKEN=apify_api_xxxxx
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import httpx

from app.core.exceptions import (
    InstagramProviderError,
    InstagramUserNotFoundError,
    RateLimitError,
)
from app.core.logging import get_logger
from app.schemas.social import Platform, PostData, RawSocialProfile
from app.services.social_base import SocialMediaProvider

logger = get_logger(__name__)

_APIFY_RUN_URL = (
    "https://api.apify.com/v2/acts/{actor_id}/run-sync-get-dataset-items"
)
_TIMEOUT = 120.0
_MAX_POSTS = 12


class FacebookApifyProvider(SocialMediaProvider):
    """Fetch Facebook profile + posts via Apify's facebook-posts-scraper."""

    def __init__(self, api_token: str, actor_id: str = "apify~facebook-posts-scraper"):
        if not api_token:
            raise ValueError("APIFY_API_TOKEN is required for the Apify Facebook provider.")
        self._token = api_token
        self._url = _APIFY_RUN_URL.format(actor_id=actor_id)

    async def get_profile(self, username: str) -> RawSocialProfile:
        username = username.lstrip("@").lower()
        logger.info("Apify Facebook: fetching @%s", username)

        params = {"token": self._token}
        payload = {
            "startUrls": [{"url": f"https://www.facebook.com/{username}"}],
            "resultsLimit": _MAX_POSTS,
        }

        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            try:
                resp = await client.post(self._url, params=params, json=payload)
            except httpx.TimeoutException:
                raise InstagramProviderError(
                    "Apify Facebook request timed out."
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
        # Filter out error items
        post_items = [
            p for p in items
            if isinstance(p, dict) and not p.get("error") and p.get("pageName")
        ]

        if not post_items:
            raise InstagramUserNotFoundError(
                f"@{username} was not found on Facebook or has no public posts."
            )

        return self._build_profile(post_items, username)

    def _build_profile(self, post_items: list[dict], username: str) -> RawSocialProfile:
        """Build a RawSocialProfile from Apify Facebook post items."""
        first = post_items[0]

        # Page/profile metadata
        page_name = first.get("pageName", "")
        user_obj = first.get("user", {})
        profile_pic = None
        if user_obj:
            profile_pic = user_obj.get("profilePic")

        facebook_url = first.get("facebookUrl", "")

        # Build posts list
        posts: list[PostData] = []
        for item in post_items[:_MAX_POSTS]:
            # Total reactions = likes (or sum of individual reactions)
            likes = item.get("likes", 0)
            if not likes:
                likes = sum([
                    item.get("reactionLikeCount", 0),
                    item.get("reactionLoveCount", 0),
                    item.get("reactionWowCount", 0),
                    item.get("reactionHahaCount", 0),
                    item.get("reactionSadCount", 0),
                    item.get("reactionAngryCount", 0),
                    item.get("reactionCareCount", 0),
                ])

            comments = item.get("comments", 0)
            text = item.get("text") or item.get("message") or ""

            ts = None
            time_str = item.get("time")
            if time_str:
                try:
                    ts = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
                except ValueError:
                    pass
            elif item.get("timestamp"):
                try:
                    ts = datetime.fromtimestamp(int(item["timestamp"]), tz=timezone.utc)
                except (ValueError, OSError):
                    pass

            has_media = bool(item.get("media"))
            posts.append(PostData(
                likes_count=int(likes or 0),
                comments_count=int(comments or 0),
                caption=text,
                is_video="video" in str(item.get("media", [])).lower(),
                media_type="Post",
                timestamp=ts,
            ))

        logger.info(
            "Apify Facebook: @%s — %s | %d posts fetched",
            username, page_name, len(posts),
        )

        return RawSocialProfile(
            platform=Platform.FACEBOOK,
            username=username,
            full_name=page_name,
            biography="",
            profile_pic_url=profile_pic,
            followers_count=0,  # Not directly in post items; will be estimated
            following_count=0,
            posts_count=len(posts),
            is_verified=False,
            is_private=False,
            recent_posts=posts,
        )
