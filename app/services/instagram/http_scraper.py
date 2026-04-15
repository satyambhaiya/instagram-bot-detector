"""
HTTP Scraper Provider — Real Instagram data, no API key, no extra dependencies.

Strategy (tries each method in order, stops at first success):
  1. Instagram internal API  — i.instagram.com/api/v1/users/web_profile_info/
  2. Legacy JSON endpoint     — www.instagram.com/{username}/?__a=1&__d=dis
  3. HTML page scraping       — extract JSON embedded in the profile page scripts

Important: a 404 from Instagram's API does NOT mean the account doesn't exist.
Instagram returns 404 to block unauthenticated API requests. We only confirm
"not found" when the actual profile HTML page returns 404.

Usage (.env)
------------
    INSTAGRAM_PROVIDER=http
"""

from __future__ import annotations

import json
import re
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
# Backward compat alias
RawInstagramProfile = RawSocialProfile
from app.services.instagram.base import InstagramProvider

logger = get_logger(__name__)
_UTC = timezone.utc
_MAX_POSTS = 12

# ── Request headers ───────────────────────────────────────────────────────────

# Mobile Safari headers used for the internal API endpoints
_API_HEADERS = {
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "User-Agent": (
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) "
        "Version/17.0 Mobile/15E148 Safari/604.1"
    ),
    "X-IG-App-ID": "936619743392459",
    "X-ASBD-ID": "129477",
    "X-IG-WWW-Claim": "0",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
    "Referer": "https://www.instagram.com/",
    "Origin": "https://www.instagram.com",
}

# Desktop Chrome headers used for HTML page scraping
_BROWSER_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Upgrade-Insecure-Requests": "1",
}

_TIMEOUT = 25.0


class HttpScraperProvider(InstagramProvider):
    """
    Multi-strategy Instagram scraper using only httpx (no extra dependencies).
    Automatically tries all available methods before giving up.
    """

    async def get_profile(self, username: str) -> RawInstagramProfile:
        logger.info("HttpScraper: fetching @%s", username)

        # ── Method 1 & 2: API endpoints (fast but often blocked) ─────────────
        async with httpx.AsyncClient(
            headers=_API_HEADERS,
            timeout=_TIMEOUT,
            follow_redirects=True,
        ) as client:
            await self._warm_cookies(client)

            result = await self._try_web_profile_info(client, username)
            if result:
                logger.info("HttpScraper: got @%s via web_profile_info", username)
                return result

            result = await self._try_a1_endpoint(client, username)
            if result:
                logger.info("HttpScraper: got @%s via __a=1 endpoint", username)
                return result

        # ── Method 3: HTML page scraping (most reliable) ──────────────────────
        result = await self._try_html_scrape(username)
        if result:
            logger.info("HttpScraper: got @%s via HTML scraping", username)
            return result

        raise InstagramProviderError(
            "Could not retrieve Instagram data. "
            "Instagram is blocking requests from this network. "
            "Set INSTAGRAM_PROVIDER=apify or INSTAGRAM_PROVIDER=rapidapi "
            "in your .env file for reliable production access."
        )

    # ── Cookie warm-up ────────────────────────────────────────────────────────

    async def _warm_cookies(self, client: httpx.AsyncClient) -> None:
        try:
            resp = await client.get("https://www.instagram.com/", timeout=8)
            csrf = client.cookies.get("csrftoken")
            if csrf:
                client.headers["X-CSRFToken"] = csrf
        except Exception:
            pass

    # ── Method 1: Internal API ────────────────────────────────────────────────

    async def _try_web_profile_info(
        self, client: httpx.AsyncClient, username: str
    ) -> Optional[RawInstagramProfile]:
        try:
            resp = await client.get(
                "https://i.instagram.com/api/v1/users/web_profile_info/",
                params={"username": username},
            )
        except httpx.RequestError as e:
            logger.debug("web_profile_info request error: %s", e)
            return None

        logger.debug("web_profile_info: HTTP %d for @%s", resp.status_code, username)

        if resp.status_code == 429:
            raise RateLimitError("Instagram rate limit. Wait 1-2 minutes.")

        # 404/401/403 here = Instagram blocking, NOT account missing — try next method
        if resp.status_code != 200:
            return None

        try:
            body = resp.json()
        except Exception:
            return None

        # Explicit "not found" signal in response body
        if body.get("message") == "user_not_found":
            raise InstagramUserNotFoundError(f"@{username} does not exist on Instagram.")

        user = (body.get("data") or {}).get("user")
        if not user:
            return None

        return self._parse_api_user(user, username)

    # ── Method 2: Legacy ?__a=1 endpoint ─────────────────────────────────────

    async def _try_a1_endpoint(
        self, client: httpx.AsyncClient, username: str
    ) -> Optional[RawInstagramProfile]:
        try:
            resp = await client.get(
                f"https://www.instagram.com/{username}/",
                params={"__a": "1", "__d": "dis"},
            )
        except httpx.RequestError:
            return None

        logger.debug("__a=1 endpoint: HTTP %d for @%s", resp.status_code, username)

        if resp.status_code == 429:
            raise RateLimitError("Instagram rate limit. Wait 1-2 minutes.")
        if resp.status_code != 200:
            return None

        try:
            body = resp.json()
        except Exception:
            return None

        user = (
            (body.get("graphql") or {}).get("user")
            or (body.get("data") or {}).get("user")
            or body.get("user")
        )
        if not user:
            return None

        return self._parse_graphql_user(user, username)

    # ── Method 3: HTML page scraping ──────────────────────────────────────────

    async def _try_html_scrape(self, username: str) -> Optional[RawInstagramProfile]:
        """
        Visit the public profile page as a regular browser and extract
        the JSON data Instagram embeds in script tags for SEO purposes.
        """
        async with httpx.AsyncClient(
            headers=_BROWSER_HEADERS,
            timeout=_TIMEOUT,
            follow_redirects=True,
        ) as client:
            try:
                resp = await client.get(f"https://www.instagram.com/{username}/")
            except httpx.RequestError as e:
                logger.debug("HTML scrape request error: %s", e)
                return None

        logger.debug("HTML page: HTTP %d for @%s", resp.status_code, username)

        # A real 404 on the HTML page = account genuinely does not exist
        if resp.status_code == 404:
            raise InstagramUserNotFoundError(f"@{username} does not exist on Instagram.")

        if resp.status_code == 429:
            raise RateLimitError("Instagram rate limit. Wait 1-2 minutes.")

        if resp.status_code != 200:
            return None

        return self._parse_html(resp.text, username)

    def _parse_html(self, html: str, username: str) -> Optional[RawInstagramProfile]:
        """
        Extract profile data from Instagram's HTML page.
        Instagram embeds JSON in <script> tags for SEO/hydration.
        """
        # ── Try to find user JSON block in script tags ─────────────────────
        # Instagram embeds profile data in multiple script blocks
        user_data = self._extract_user_json(html, username)

        if user_data:
            return self._parse_api_user(user_data, username)

        # ── Fallback: extract individual fields with regex ─────────────────
        profile = self._extract_fields_regex(html, username)
        if profile:
            return profile

        # Account page loaded (200 OK) but we couldn't parse the data
        # Return a minimal profile — at least we know the account exists
        logger.warning(
            "HTML parse for @%s returned 200 but no parseable data. "
            "Instagram likely changed its HTML format. "
            "Using minimal profile.", username
        )
        return RawInstagramProfile(platform=Platform.INSTAGRAM, username=username)

    def _extract_user_json(self, html: str, username: str) -> Optional[dict]:
        """Find and parse the user JSON object from embedded script data."""
        # Try to extract JSON from <script type="application/json"> blocks
        script_contents = re.findall(
            r'<script[^>]*type=["\']application/json["\'][^>]*>(.*?)</script>',
            html, re.DOTALL
        )

        for content in script_contents:
            try:
                data = json.loads(content)
                user = self._deep_find_user(data, username)
                if user and isinstance(user, dict) and user.get("edge_followed_by"):
                    return user
            except (json.JSONDecodeError, RecursionError):
                continue

        # Also search all <script> blocks for inline JSON containing user data
        all_scripts = re.findall(r'<script[^>]*>(.*?)</script>', html, re.DOTALL)
        for script in all_scripts:
            # Look for the window._sharedData pattern (older Instagram)
            match = re.search(r'window\._sharedData\s*=\s*({.+?});\s*</script>', script + "</script>", re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(1))
                    pages = (data.get("entry_data") or {}).get("ProfilePage", [])
                    if pages:
                        return (pages[0].get("graphql") or {}).get("user")
                except Exception:
                    pass

        return None

    def _deep_find_user(self, data, username: str, depth: int = 0) -> Optional[dict]:
        """Recursively search JSON for a user object matching the username."""
        if depth > 8:
            return None
        if isinstance(data, dict):
            if data.get("username") == username and "edge_followed_by" in data:
                return data
            for v in data.values():
                result = self._deep_find_user(v, username, depth + 1)
                if result:
                    return result
        elif isinstance(data, list):
            for item in data[:20]:  # limit list traversal
                result = self._deep_find_user(item, username, depth + 1)
                if result:
                    return result
        return None

    def _extract_fields_regex(self, html: str, username: str) -> Optional[RawInstagramProfile]:
        """
        Last resort: extract individual fields using regex on the raw HTML.
        Works when Instagram embeds profile data as JSON literals in script tags.
        """
        def find_int(pattern: str) -> int:
            m = re.search(pattern, html)
            return int(m.group(1)) if m else 0

        def find_str(pattern: str) -> str:
            m = re.search(pattern, html)
            if not m:
                return ""
            return self._unescape(m.group(1))

        def find_bool(pattern: str) -> bool:
            m = re.search(pattern, html)
            return m.group(1) == "true" if m else False

        followers = find_int(r'"edge_followed_by":\{"count":(\d+)\}')
        following = find_int(r'"edge_follow":\{"count":(\d+)\}')
        posts_count = find_int(r'"edge_owner_to_timeline_media":\{"count":(\d+)')
        full_name = find_str(r'"full_name":"((?:[^"\\]|\\.)*)"')
        biography = find_str(r'"biography":"((?:[^"\\]|\\.)*)"')
        is_verified = find_bool(r'"is_verified":(true|false)')
        is_private = find_bool(r'"is_private":(true|false)')
        is_business = find_bool(r'"is_business_account":(true|false)')
        profile_pic = find_str(r'"profile_pic_url(?:_hd)?":"(https?://[^"]+)"')
        external_url = find_str(r'"external_url":"(https?://[^"]*)"')

        # Only proceed if we found at least some key data
        if followers == 0 and following == 0 and not full_name and not biography:
            return None

        # Extract recent posts
        posts = self._extract_posts_regex(html)

        return RawInstagramProfile(
            platform=Platform.INSTAGRAM,
            username=username,
            full_name=full_name,
            biography=biography,
            profile_pic_url=profile_pic or None,
            followers_count=followers,
            following_count=following,
            posts_count=posts_count,
            is_verified=is_verified,
            is_private=is_private,
            is_business=is_business,
            external_url=external_url or None,
            recent_posts=posts,
        )

    def _extract_posts_regex(self, html: str) -> list[PostData]:
        """Extract post data from HTML using regex."""
        posts: list[PostData] = []
        # Find all taken_at_timestamp occurrences near like/comment counts
        blocks = re.findall(
            r'"taken_at_timestamp":(\d+).*?"edge_liked_by":\{"count":(\d+)\}.*?"edge_media_to_comment":\{"count":(\d+)\}',
            html
        )
        for ts_str, likes_str, comments_str in blocks[:_MAX_POSTS]:
            try:
                ts = datetime.fromtimestamp(int(ts_str), tz=_UTC)
                posts.append(PostData(
                    likes_count=int(likes_str),
                    comments_count=int(comments_str),
                    timestamp=ts,
                ))
            except (ValueError, OSError):
                continue
        return posts

    # ── Parsers ───────────────────────────────────────────────────────────────

    def _parse_api_user(self, user: dict, username: str) -> RawInstagramProfile:
        media_node = user.get("edge_owner_to_timeline_media") or {}
        posts = self._parse_post_edges(media_node.get("edges", []))
        return RawInstagramProfile(
            platform=Platform.INSTAGRAM,
            username=user.get("username") or username,
            full_name=user.get("full_name", ""),
            biography=user.get("biography", ""),
            profile_pic_url=user.get("profile_pic_url_hd") or user.get("profile_pic_url"),
            followers_count=int((user.get("edge_followed_by") or {}).get("count", 0)),
            following_count=int((user.get("edge_follow") or {}).get("count", 0)),
            posts_count=int(media_node.get("count", 0)),
            is_verified=bool(user.get("is_verified", False)),
            is_private=bool(user.get("is_private", False)),
            is_business=bool(user.get("is_business_account", False)),
            external_url=user.get("external_url") or None,
            recent_posts=posts,
        )

    def _parse_graphql_user(self, user: dict, username: str) -> RawInstagramProfile:
        media_node = user.get("edge_owner_to_timeline_media") or {}
        posts = self._parse_post_edges(media_node.get("edges", []))
        followers = (user.get("edge_followed_by") or {}).get("count", 0)
        following = (user.get("edge_follow") or {}).get("count", 0)
        return RawInstagramProfile(
            platform=Platform.INSTAGRAM,
            username=user.get("username") or username,
            full_name=user.get("full_name", ""),
            biography=user.get("biography", ""),
            profile_pic_url=user.get("profile_pic_url_hd") or user.get("profile_pic_url"),
            followers_count=int(followers),
            following_count=int(following),
            posts_count=int(media_node.get("count", 0)),
            is_verified=bool(user.get("is_verified", False)),
            is_private=bool(user.get("is_private", False)),
            is_business=bool(user.get("is_business_account", False)),
            external_url=user.get("external_url") or None,
            recent_posts=posts,
        )

    def _parse_post_edges(self, edges: list) -> list[PostData]:
        posts: list[PostData] = []
        for edge in edges[:_MAX_POSTS]:
            node = edge.get("node") or edge
            if not node:
                continue
            ts = None
            raw_ts = node.get("taken_at_timestamp") or node.get("taken_at")
            if isinstance(raw_ts, int):
                try:
                    ts = datetime.fromtimestamp(raw_ts, tz=_UTC)
                except (ValueError, OSError):
                    pass
            caption = ""
            cap_edges = (node.get("edge_media_to_caption") or {}).get("edges", [])
            if cap_edges:
                caption = (cap_edges[0].get("node") or {}).get("text", "")
            likes = (
                (node.get("edge_liked_by") or {}).get("count")
                or (node.get("edge_media_preview_like") or {}).get("count")
                or node.get("like_count", 0)
            )
            comments = (
                (node.get("edge_media_to_comment") or {}).get("count")
                or node.get("comment_count", 0)
            )
            is_video = bool(node.get("is_video") or node.get("__typename") == "GraphVideo")
            posts.append(PostData(
                likes_count=int(likes or 0),
                comments_count=int(comments or 0),
                caption=caption,
                is_video=is_video,
                media_type="Video" if is_video else "Image",
                timestamp=ts,
            ))
        return posts

    # ── Utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def _unescape(s: str) -> str:
        """Decode Unicode escapes (\\u00e9 → é) and fix HTML entities."""
        try:
            return s.encode("utf-8").decode("unicode_escape").encode("latin-1").decode("utf-8")
        except Exception:
            return s.replace("\\u0026", "&").replace("\\u003c", "<").replace("\\u003e", ">")
