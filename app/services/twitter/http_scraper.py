"""
Twitter/X HTTP Scraper — fetches public profile data without API keys.

Strategy (ordered by reliability):
  1. Twitter GraphQL API with guest token — full profile data (name, bio,
     followers, following, tweets, verified, created_at, profile pic)
  2. Nitter instances — open-source Twitter frontend
  3. Direct x.com meta tag scraping — limited fallback

Note: The GraphQL approach uses Twitter's own internal API with a guest
token, the same mechanism the website uses for logged-out views.
"""

from __future__ import annotations

import json
import re
import urllib.parse
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
_UTC = timezone.utc
_TIMEOUT = 15.0
_MAX_POSTS = 12

# Twitter's public bearer token (used by the web app for unauthenticated requests)
_BEARER = "AAAAAAAAAAAAAAAAAAAAANRILgAAAAAAnNwIzUejRCOuH5E6I8xnZz4puTs%3D1Zv7ttfk8LF81IUq16cHjhLTvJu4FA33AGWWjCpTnA"

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# GraphQL query features (required by the API)
_GQL_FEATURES = {
    "hidden_profile_subscriptions_enabled": True,
    "rweb_tipjar_consumption_enabled": True,
    "responsive_web_graphql_exclude_directive_enabled": True,
    "verified_phone_label_enabled": False,
    "subscriptions_verification_info_is_identity_verified_enabled": True,
    "subscriptions_verification_info_verified_since_enabled": True,
    "highlights_tweets_tab_ui_enabled": True,
    "responsive_web_twitter_article_notes_tab_enabled": True,
    "subscriptions_feature_can_gift_premium": True,
    "creator_subscriptions_tweet_preview_api_enabled": True,
    "responsive_web_graphql_skip_user_profile_image_extensions_enabled": False,
    "responsive_web_graphql_timeline_navigation_enabled": True,
}

_NITTER_INSTANCES = [
    "https://nitter.poast.org",
    "https://nitter.privacydev.net",
    "https://nitter.net",
]


class TwitterHttpScraper(SocialMediaProvider):
    """Scrape public Twitter/X profile data via multiple fallback methods."""

    async def get_profile(self, username: str) -> RawSocialProfile:
        username = username.lstrip("@").lower()

        # 1. Try GraphQL API with guest token (most reliable)
        profile = await self._try_graphql_api(username)
        if profile:
            return profile

        # 2. Try nitter instances
        profile = await self._try_nitter(username)
        if profile:
            return profile

        # 3. Try direct x.com meta scraping
        profile = await self._try_xcom_scrape(username)
        if profile:
            return profile

        raise InstagramUserNotFoundError(
            f"Could not fetch Twitter profile for @{username}. "
            "The account may not exist or Twitter is blocking requests."
        )

    # ── GraphQL API ──────────────────────────────────────────────────────────

    async def _try_graphql_api(self, username: str) -> Optional[RawSocialProfile]:
        """Fetch profile via Twitter's internal GraphQL API with a guest token."""
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True) as client:
                headers = {
                    "User-Agent": _HEADERS["User-Agent"],
                    "Authorization": f"Bearer {_BEARER}",
                }

                # Step 1: obtain a guest token
                resp = await client.post(
                    "https://api.twitter.com/1.1/guest/activate.json",
                    headers=headers,
                )
                if resp.status_code != 200:
                    logger.debug("Guest token request failed: %s", resp.status_code)
                    return None

                guest_token = resp.json().get("guest_token")
                if not guest_token:
                    return None

                headers["x-guest-token"] = guest_token

                # Step 2: call UserByScreenName GraphQL endpoint
                variables = json.dumps(
                    {"screen_name": username, "withSafetyModeUserFields": True}
                )
                features = json.dumps(_GQL_FEATURES)
                params = urllib.parse.urlencode(
                    {"variables": variables, "features": features}
                )

                url = f"https://twitter.com/i/api/graphql/xc8f1g7BYqr6VTzTbvNlGw/UserByScreenName?{params}"
                resp = await client.get(url, headers=headers)

                if resp.status_code == 429:
                    logger.warning("Twitter GraphQL rate limited")
                    return None
                if resp.status_code != 200:
                    logger.debug("GraphQL API returned %s for @%s", resp.status_code, username)
                    return None

                return self._parse_graphql_response(resp.json(), username)
        except Exception as e:
            logger.debug("GraphQL API failed for @%s: %s", username, e)
            return None

    def _parse_graphql_response(
        self, data: dict, username: str
    ) -> Optional[RawSocialProfile]:
        """Parse the GraphQL UserByScreenName response."""
        user = data.get("data", {}).get("user", {}).get("result", {})
        if not user:
            return None

        # Check for suspended/not found
        typename = user.get("__typename", "")
        if typename == "UserUnavailable":
            return None

        legacy = user.get("legacy", {})
        if not legacy:
            return None

        full_name = legacy.get("name", "")
        bio = legacy.get("description", "")
        followers = legacy.get("followers_count", 0)
        following = legacy.get("friends_count", 0)
        statuses = legacy.get("statuses_count", 0)
        is_verified = user.get("is_blue_verified", False) or legacy.get("verified", False)
        profile_pic = legacy.get("profile_image_url_https", "")
        is_protected = legacy.get("protected", False)

        # Parse account creation date
        created_at = None
        created_str = legacy.get("created_at", "")
        if created_str:
            try:
                created_at = datetime.strptime(
                    created_str, "%a %b %d %H:%M:%S %z %Y"
                )
            except ValueError:
                pass

        # Replace _normal with _400x400 for better quality pic
        if profile_pic:
            profile_pic = profile_pic.replace("_normal.", "_400x400.")

        logger.info(
            "GraphQL: @%s — %s | %d followers | %d tweets",
            username, full_name, followers, statuses,
        )

        return RawSocialProfile(
            platform=Platform.TWITTER,
            username=username,
            full_name=full_name,
            biography=bio,
            profile_pic_url=profile_pic or None,
            followers_count=followers,
            following_count=following,
            posts_count=statuses,
            is_verified=is_verified,
            is_private=is_protected,
            recent_posts=[],
            account_created_at=created_at,
        )

    # ── Nitter fallback ──────────────────────────────────────────────────────

    async def _try_nitter(self, username: str) -> Optional[RawSocialProfile]:
        """Try scraping from Nitter (open-source Twitter frontend)."""
        for instance in _NITTER_INSTANCES:
            try:
                url = f"{instance}/{username}"
                async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True) as client:
                    resp = await client.get(url, headers=_HEADERS)
                    if resp.status_code == 404:
                        raise InstagramUserNotFoundError(f"@{username} not found on Twitter")
                    if resp.status_code != 200:
                        continue

                    profile = self._parse_nitter_html(resp.text, username)
                    if profile:
                        logger.info("Nitter (%s): fetched @%s", instance, username)
                        return profile
            except InstagramUserNotFoundError:
                raise
            except Exception as e:
                logger.debug("Nitter %s failed: %s", instance, e)
                continue
        return None

    def _parse_nitter_html(self, html: str, username: str) -> Optional[RawSocialProfile]:
        """Parse Nitter HTML to extract profile data."""
        followers = self._extract_nitter_stat(html, "followers")
        following = self._extract_nitter_stat(html, "following")
        tweets_count = (
            self._extract_nitter_stat(html, "tweets")
            or self._extract_nitter_stat(html, "posts")
        )

        bio_match = re.search(r'<p[^>]*class="bio"[^>]*>(.*?)</p>', html, re.DOTALL)
        bio = re.sub(r"<[^>]+>", "", bio_match.group(1)).strip() if bio_match else ""

        name_match = re.search(
            r'<a[^>]*class="profile-card-fullname"[^>]*>(.*?)</a>', html, re.DOTALL
        )
        full_name = (
            re.sub(r"<[^>]+>", "", name_match.group(1)).strip()
            if name_match
            else username
        )

        is_verified = "icon-ok" in html or "verified" in html.lower()

        pic_match = re.search(
            r'<img[^>]*class="profile-card-avatar"[^>]*src="([^"]+)"', html
        )
        profile_pic = pic_match.group(1) if pic_match else None

        posts = self._extract_nitter_tweets(html)

        if followers is None and following is None and not posts:
            return None

        return RawSocialProfile(
            platform=Platform.TWITTER,
            username=username,
            full_name=full_name,
            biography=bio,
            profile_pic_url=profile_pic,
            followers_count=followers or 0,
            following_count=following or 0,
            posts_count=tweets_count or 0,
            is_verified=is_verified,
            is_private=False,
            recent_posts=posts,
        )

    def _extract_nitter_stat(self, html: str, stat_name: str) -> Optional[int]:
        pattern = rf'<span[^>]*class="profile-stat-num"[^>]*>([\d,.KkMm]+)</span>\s*<span[^>]*>{stat_name}</span>'
        match = re.search(pattern, html, re.IGNORECASE)
        if not match:
            pattern2 = rf"{stat_name}[^<]*<[^>]*>([\d,.KkMm]+)</span>"
            match = re.search(pattern2, html, re.IGNORECASE)
        if match:
            return self._parse_count(match.group(1))
        return None

    def _extract_nitter_tweets(self, html: str) -> list[PostData]:
        posts = []
        tweet_blocks = re.findall(
            r'<div[^>]*class="tweet-content[^"]*"[^>]*>(.*?)</div>',
            html,
            re.DOTALL,
        )
        for block in tweet_blocks[:_MAX_POSTS]:
            text = re.sub(r"<[^>]+>", "", block).strip()
            if text:
                posts.append(
                    PostData(likes_count=0, comments_count=0, caption=text, media_type="Tweet")
                )
        return posts

    # ── x.com meta fallback ──────────────────────────────────────────────────

    async def _try_xcom_scrape(self, username: str) -> Optional[RawSocialProfile]:
        url = f"https://x.com/{username}"
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True) as client:
                resp = await client.get(url, headers=_HEADERS)
                if resp.status_code != 200:
                    return None
                return self._parse_xcom_html(resp.text, username)
        except Exception as e:
            logger.debug("x.com scrape failed for @%s: %s", username, e)
            return None

    def _parse_xcom_html(self, html: str, username: str) -> Optional[RawSocialProfile]:
        desc_match = re.search(
            r'<meta[^>]*name="description"[^>]*content="([^"]*)"', html
        )
        title_match = re.search(
            r'<meta[^>]*property="og:title"[^>]*content="([^"]*)"', html
        )

        full_name = ""
        if title_match:
            name_text = title_match.group(1)
            name_match = re.match(r"(.*?)\s*\(@", name_text)
            if name_match:
                full_name = name_match.group(1).strip()

        bio = ""
        followers = 0
        following = 0
        if desc_match:
            desc = desc_match.group(1)
            bio = desc
            f_match = re.search(r"([\d,.]+[KkMm]?)\s*Followers", desc)
            fg_match = re.search(r"([\d,.]+[KkMm]?)\s*Following", desc)
            if f_match:
                followers = self._parse_count(f_match.group(1))
            if fg_match:
                following = self._parse_count(fg_match.group(1))

        if not full_name and not followers:
            return None

        return RawSocialProfile(
            platform=Platform.TWITTER,
            username=username,
            full_name=full_name,
            biography=bio,
            followers_count=followers,
            following_count=following,
            recent_posts=[],
        )

    # ── Utilities ────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_count(text: str) -> int:
        text = text.strip().replace(",", "")
        multiplier = 1
        if text.upper().endswith("K"):
            multiplier = 1_000
            text = text[:-1]
        elif text.upper().endswith("M"):
            multiplier = 1_000_000
            text = text[:-1]
        elif text.upper().endswith("B"):
            multiplier = 1_000_000_000
            text = text[:-1]
        try:
            return int(float(text) * multiplier)
        except ValueError:
            return 0
