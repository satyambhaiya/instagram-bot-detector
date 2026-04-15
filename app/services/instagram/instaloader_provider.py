"""
InstaLoader Provider — Real Instagram data, no API key required.

Uses the `instaloader` library to fetch public profile data directly from
Instagram. Since instaloader is synchronous, network calls are executed in a
thread-pool executor so they don't block the FastAPI event loop.

Limitations
-----------
- Private accounts: only profile-level data (no posts).
- Instagram may rate-limit aggressive usage. For production at scale, prefer
  Apify or RapidAPI (paid). For a student project / demo, this is sufficient.
- Instagram occasionally blocks scraping from cloud IPs. Running locally is
  more reliable.

Usage (.env)
------------
    INSTAGRAM_PROVIDER=instaloader
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import timezone

import instaloader
import instaloader.exceptions

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

_UTC = timezone.utc
_MAX_POSTS = 12   # number of recent posts to fetch


class InstaLoaderProvider(InstagramProvider):
    """Fetches real Instagram data using instaloader (no API key needed)."""

    def __init__(self):
        # Configure instaloader to skip all downloads — we only want metadata
        self._loader = instaloader.Instaloader(
            download_pictures=False,
            download_videos=False,
            download_video_thumbnails=False,
            download_geotags=False,
            download_comments=False,
            save_metadata=False,
            compress_json=False,
            quiet=True,
            # Respect Instagram's rate limits
            sleep=True,
            max_connection_attempts=3,
        )
        # Single worker: Instagram dislikes concurrent scraping from the same IP
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="instaloader")

    async def get_profile(self, username: str) -> RawInstagramProfile:
        logger.info("InstaLoader: fetching real profile for @%s", username)
        loop = asyncio.get_event_loop()
        # Run synchronous instaloader code in a thread so we don't block the event loop
        return await loop.run_in_executor(self._executor, self._fetch_sync, username)

    # ── Synchronous fetch (runs in thread pool) ───────────────────────────────

    def _fetch_sync(self, username: str) -> RawInstagramProfile:
        try:
            profile = instaloader.Profile.from_username(self._loader.context, username)
        except instaloader.exceptions.ProfileNotExistsException:
            raise InstagramUserNotFoundError(f"@{username} does not exist on Instagram.")
        except instaloader.exceptions.ConnectionException as exc:
            err = str(exc).lower()
            if "429" in err or "rate" in err or "too many" in err:
                raise RateLimitError(
                    "Instagram is rate-limiting requests. Please wait a few minutes."
                )
            raise InstagramProviderError(f"Connection error while fetching @{username}: {exc}")
        except Exception as exc:
            raise InstagramProviderError(f"Unexpected error fetching @{username}: {exc}")

        # ── Fetch recent posts ────────────────────────────────────────────────
        posts: list[PostData] = []

        if not profile.is_private:
            try:
                for post in profile.get_posts():
                    if len(posts) >= _MAX_POSTS:
                        break
                    ts = post.date_utc.replace(tzinfo=_UTC) if post.date_utc else None
                    posts.append(PostData(
                        likes_count=post.likes,
                        comments_count=post.comments,
                        caption=post.caption or "",
                        is_video=post.is_video,
                        media_type="Video" if post.is_video else "Image",
                        timestamp=ts,
                    ))
            except instaloader.exceptions.PrivateProfileNotFollowedException:
                # Account turned private between profile fetch and posts fetch
                logger.warning("@%s posts are private, skipping post fetch.", username)
            except Exception as exc:
                # Posts are non-critical — log and continue with empty list
                logger.warning("Could not fetch posts for @%s: %s", username, exc)
        else:
            logger.info("@%s is private — skipping post fetch.", username)

        return RawInstagramProfile(
            username=profile.username,
            full_name=profile.full_name or "",
            biography=profile.biography or "",
            profile_pic_url=profile.profile_pic_url,
            followers_count=profile.followers,
            following_count=profile.followees,
            posts_count=profile.mediacount,
            is_verified=profile.is_verified,
            is_private=profile.is_private,
            is_business=profile.is_business_account,
            external_url=profile.external_url or None,
            recent_posts=posts,
        )
