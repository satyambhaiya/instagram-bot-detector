"""
RapidAPI — Instagram Scraper API2 provider.

Requires: RAPIDAPI_KEY in .env
Host:     instagram-scraper-api2.p.rapidapi.com

Subscribe at: https://rapidapi.com/search/instagram-scraper
"""

from __future__ import annotations

from datetime import datetime, timezone

import httpx

from app.core.exceptions import (
    InstagramPrivateAccountError,
    InstagramProviderError,
    InstagramUserNotFoundError,
    RateLimitError,
)
from app.core.logging import get_logger
from app.schemas.instagram import PostData, RawInstagramProfile
from app.services.instagram.base import InstagramProvider

logger = get_logger(__name__)

_BASE = "https://{host}"
_TIMEOUT = 30.0


class RapidAPIProvider(InstagramProvider):
    """
    Fetches profile info + recent posts via RapidAPI's Instagram Scraper API2.
    Makes two sequential requests: one for profile info, one for posts.
    """

    def __init__(self, api_key: str, host: str = "instagram-scraper-api2.p.rapidapi.com"):
        if not api_key:
            raise ValueError("RAPIDAPI_KEY is required for the RapidAPI provider.")
        self._headers = {
            "X-RapidAPI-Key": api_key,
            "X-RapidAPI-Host": host,
        }
        self._base = _BASE.format(host=host)

    async def get_profile(self, username: str) -> RawInstagramProfile:
        logger.info("RapidAPI: fetching profile for @%s", username)

        async with httpx.AsyncClient(
            base_url=self._base,
            headers=self._headers,
            timeout=_TIMEOUT,
        ) as client:
            profile_data = await self._fetch_info(client, username)
            posts = await self._fetch_posts(client, username)

        return self._build_profile(profile_data, posts)

    # ── Private helpers ───────────────────────────────────────────────────────

    async def _fetch_info(self, client: httpx.AsyncClient, username: str) -> dict:
        try:
            resp = await client.get("/v1/info", params={"username_or_id_or_url": username})
        except httpx.RequestError as exc:
            raise InstagramProviderError(f"Network error: {exc}")

        self._raise_for_status(resp, username)
        body = resp.json()
        data = body.get("data") or body
        if not data or data.get("pk") is None and data.get("id") is None:
            raise InstagramUserNotFoundError(f"@{username} was not found on Instagram.")
        return data

    async def _fetch_posts(self, client: httpx.AsyncClient, username: str) -> list[dict]:
        try:
            resp = await client.get("/v1/posts", params={"username_or_id_or_url": username})
        except httpx.RequestError:
            return []   # posts are optional — degrade gracefully

        if resp.status_code != 200:
            return []

        body = resp.json()
        items = (
            (body.get("data") or {}).get("items")
            or body.get("items")
            or []
        )
        return items

    def _raise_for_status(self, resp: httpx.Response, username: str) -> None:
        if resp.status_code == 404:
            raise InstagramUserNotFoundError(f"@{username} was not found on Instagram.")
        if resp.status_code == 429:
            raise RateLimitError("RapidAPI rate limit reached.")
        if resp.status_code in (401, 403):
            raise InstagramProviderError("Invalid RapidAPI key.")
        if resp.status_code != 200:
            raise InstagramProviderError(
                f"RapidAPI returned HTTP {resp.status_code}: {resp.text[:200]}"
            )

    def _build_profile(self, data: dict, raw_posts: list[dict]) -> RawInstagramProfile:
        followers = (
            (data.get("edge_followed_by") or {}).get("count")
            or data.get("follower_count")
            or 0
        )
        following = (
            (data.get("edge_follow") or {}).get("count")
            or data.get("following_count")
            or 0
        )
        media_count = (
            (data.get("edge_owner_to_timeline_media") or {}).get("count")
            or data.get("media_count")
            or 0
        )

        posts: list[PostData] = []
        for p in raw_posts[:12]:
            ts = None
            raw_ts = p.get("taken_at")
            if isinstance(raw_ts, int):
                ts = datetime.fromtimestamp(raw_ts, tz=timezone.utc)
            elif isinstance(raw_ts, str):
                try:
                    ts = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
                except ValueError:
                    pass

            caption_text = ""
            cap = p.get("caption")
            if isinstance(cap, dict):
                caption_text = cap.get("text", "")
            elif isinstance(cap, str):
                caption_text = cap

            media_type_int = p.get("media_type", 1)
            is_video = media_type_int == 2 or p.get("is_video", False)

            posts.append(PostData(
                likes_count=p.get("like_count", 0),
                comments_count=p.get("comment_count", 0),
                caption=caption_text,
                is_video=is_video,
                media_type="Video" if is_video else "Image",
                timestamp=ts,
            ))

        return RawInstagramProfile(
            username=data.get("username", ""),
            full_name=data.get("full_name", ""),
            biography=data.get("biography", "") or data.get("bio", ""),
            profile_pic_url=data.get("profile_pic_url_hd") or data.get("profile_pic_url"),
            followers_count=int(followers),
            following_count=int(following),
            posts_count=int(media_count),
            is_verified=bool(data.get("is_verified", False)),
            is_private=bool(data.get("is_private", False)),
            is_business=bool(data.get("is_business_account", False)),
            external_url=data.get("external_url") or None,
            recent_posts=posts,
        )
