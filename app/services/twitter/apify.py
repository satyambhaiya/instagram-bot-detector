"""
Apify Twitter/X provider — fetches profile + recent tweets via Apify actor.

Actor:  quacker/twitter-url-scraper
Docs:   https://apify.com/quacker/twitter-url-scraper

Returns full tweet data: text, likes, retweets, replies, timestamps,
plus the embedded user object (followers, following, bio, etc.).

Usage (.env):
    TWITTER_PROVIDER=apify
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
_TIMEOUT = 90.0
_MAX_TWEETS = 20


class TwitterApifyProvider(SocialMediaProvider):
    """Fetch Twitter profile + tweets via Apify's quacker/twitter-url-scraper."""

    def __init__(self, api_token: str, actor_id: str = "quacker~twitter-url-scraper"):
        if not api_token:
            raise ValueError("APIFY_API_TOKEN is required for the Apify Twitter provider.")
        self._token = api_token
        self._url = _APIFY_RUN_URL.format(actor_id=actor_id)

    async def get_profile(self, username: str) -> RawSocialProfile:
        username = username.lstrip("@").lower()
        logger.info("Apify Twitter: fetching @%s", username)

        params = {"token": self._token}
        payload = {
            "startUrls": [{"url": f"https://twitter.com/{username}"}],
            "maxTweets": _MAX_TWEETS,
            "addUserInfo": True,
        }

        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            try:
                resp = await client.post(self._url, params=params, json=payload)
            except httpx.TimeoutException:
                raise InstagramProviderError(
                    "Apify Twitter request timed out."
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
        # Filter out error items and noResults placeholders
        tweets = [
            t for t in items
            if isinstance(t, dict) and not t.get("error") and not t.get("noResults")
        ]

        if not tweets:
            raise InstagramUserNotFoundError(
                f"@{username} was not found on Twitter or has no public tweets."
            )

        return self._build_profile(tweets, username)

    def _build_profile(self, tweets: list[dict], username: str) -> RawSocialProfile:
        """Build a RawSocialProfile from Apify tweet items."""
        # Extract user info from the first tweet's embedded user object
        user = tweets[0].get("user", {})

        full_name = user.get("name", "")
        bio = user.get("description", "")
        followers = user.get("followers_count", 0)
        following = user.get("friends_count", 0)
        statuses = user.get("statuses_count", 0)
        is_verified = user.get("is_blue_verified", False) or user.get("verified", False)
        is_protected = user.get("protected", False)
        profile_pic = user.get("profile_image_url_https", "")

        # Parse account creation date
        created_at = self._parse_twitter_date(user.get("created_at", ""))

        # Better quality profile pic
        if profile_pic:
            profile_pic = profile_pic.replace("_normal.", "_400x400.")

        # Build post list from tweets (skip pure retweets for engagement analysis)
        posts: list[PostData] = []
        for tw in tweets[:_MAX_TWEETS]:
            full_text = tw.get("full_text") or tw.get("text", "")
            # Skip retweets — they don't reflect the user's own engagement
            if full_text.startswith("RT @"):
                continue

            ts = self._parse_twitter_date(tw.get("created_at", ""))
            likes = tw.get("favorite_count", 0)
            replies = tw.get("reply_count", 0)
            has_media = bool(tw.get("entities", {}).get("media"))

            posts.append(PostData(
                likes_count=likes,
                comments_count=replies,
                caption=full_text,
                is_video=has_media,
                media_type="Tweet",
                timestamp=ts,
            ))

        logger.info(
            "Apify Twitter: @%s — %s | %d followers | %d tweets | %d posts fetched",
            username, full_name, followers, statuses, len(posts),
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
            recent_posts=posts,
            account_created_at=created_at,
        )

    @staticmethod
    def _parse_twitter_date(date_str: str) -> Optional[datetime]:
        """Parse Twitter's date format: 'Wed Dec 19 20:20:32 +0000 2007'."""
        if not date_str:
            return None
        try:
            return datetime.strptime(date_str, "%a %b %d %H:%M:%S %z %Y")
        except ValueError:
            return None
