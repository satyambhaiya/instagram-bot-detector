"""
Facebook HTTP Scraper — fetches public profile data without API keys.

Strategy:
  1. m.facebook.com (mobile) — returns og:meta tags with name, description,
     image, and sometimes likes/followers count in the description.
     Uses mobile User-Agent to avoid login redirects.
  2. touch.facebook.com — alternative mobile frontend.

Note: Facebook aggressively blocks scraping. mbasic.facebook.com redirects
to login. m.facebook.com works better with a mobile User-Agent. Data is
limited to: display name, bio/about, profile pic, and page likes count
(when available in the og:description).
"""

from __future__ import annotations

import re
from datetime import timezone
from typing import Optional

import httpx

from app.core.exceptions import (
    InstagramPrivateAccountError,
    InstagramProviderError,
    InstagramUserNotFoundError,
)
from app.core.logging import get_logger
from app.schemas.social import Platform, PostData, RawSocialProfile
from app.services.social_base import SocialMediaProvider

logger = get_logger(__name__)
_UTC = timezone.utc
_TIMEOUT = 15.0

# Mobile user-agent avoids login redirects on m.facebook.com
_MOBILE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

_DESKTOP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


class FacebookHttpScraper(SocialMediaProvider):
    """Scrape public Facebook profile data."""

    async def get_profile(self, username: str) -> RawSocialProfile:
        username = username.lstrip("@").lower()

        # Try m.facebook.com with mobile UA (best success rate)
        profile = await self._try_mobile_facebook(username)
        if profile:
            return profile

        # Try touch.facebook.com
        profile = await self._try_touch_facebook(username)
        if profile:
            return profile

        # Try www.facebook.com with desktop UA
        profile = await self._try_www_facebook(username)
        if profile:
            return profile

        raise InstagramUserNotFoundError(
            f"Could not fetch Facebook profile for '{username}'. "
            "The profile may not exist, be private, or Facebook is blocking requests."
        )

    async def _try_mobile_facebook(self, username: str) -> Optional[RawSocialProfile]:
        """Try m.facebook.com with mobile User-Agent."""
        url = f"https://m.facebook.com/{username}"
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True) as client:
                resp = await client.get(url, headers=_MOBILE_HEADERS)

                if resp.status_code == 404:
                    return None
                if resp.status_code != 200:
                    return None

                # Check for login redirect
                if "login" in str(resp.url).lower():
                    logger.debug("m.facebook.com redirected to login for @%s", username)
                    return None

                html = resp.text
                return self._parse_facebook_html(html, username, "m.facebook.com")
        except Exception as e:
            logger.debug("m.facebook.com failed for @%s: %s", username, e)
            return None

    async def _try_touch_facebook(self, username: str) -> Optional[RawSocialProfile]:
        """Try touch.facebook.com."""
        url = f"https://touch.facebook.com/{username}"
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True) as client:
                resp = await client.get(url, headers=_MOBILE_HEADERS)

                if resp.status_code != 200:
                    return None
                if "login" in str(resp.url).lower():
                    return None

                return self._parse_facebook_html(resp.text, username, "touch.facebook.com")
        except Exception as e:
            logger.debug("touch.facebook.com failed for @%s: %s", username, e)
            return None

    async def _try_www_facebook(self, username: str) -> Optional[RawSocialProfile]:
        """Try www.facebook.com with desktop UA."""
        url = f"https://www.facebook.com/{username}"
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True) as client:
                resp = await client.get(url, headers=_DESKTOP_HEADERS)

                if resp.status_code != 200:
                    return None

                return self._parse_facebook_html(resp.text, username, "www.facebook.com")
        except Exception as e:
            logger.debug("www.facebook.com failed for @%s: %s", username, e)
            return None

    # ── Parsing ──────────────────────────────────────────────────────────────

    def _parse_facebook_html(
        self, html: str, username: str, source: str
    ) -> Optional[RawSocialProfile]:
        """Parse Facebook HTML (works for m., touch., and www. variants)."""

        # og:title — display name
        title_match = re.search(
            r'<meta[^>]*property="og:title"[^>]*content="([^"]*)"', html
        )
        # Also try <title> tag
        if not title_match:
            title_match = re.search(r"<title>(.*?)</title>", html)

        full_name = ""
        if title_match:
            full_name = title_match.group(1).strip()
            # Remove " | Facebook" or " - Facebook" suffix
            full_name = re.sub(r"\s*[|–\-]\s*Facebook\s*$", "", full_name)

        # og:description — may contain likes/followers count
        desc_match = re.search(
            r'<meta[^>]*property="og:description"[^>]*content="([^"]*)"', html
        )
        bio = ""
        followers = 0
        if desc_match:
            desc = desc_match.group(1).strip()
            bio = desc

            # Extract likes count: "121,158,166 likes"
            likes_match = re.search(r"([\d,.]+)\s*likes?", desc, re.IGNORECASE)
            if likes_match:
                followers = self._parse_count(likes_match.group(1))

            # Extract followers count: "123,456 followers"
            followers_match = re.search(
                r"([\d,.]+)\s*followers?", desc, re.IGNORECASE
            )
            if followers_match:
                followers = self._parse_count(followers_match.group(1))

        # og:image — profile picture
        img_match = re.search(
            r'<meta[^>]*property="og:image"[^>]*content="([^"]*)"', html
        )
        profile_pic = img_match.group(1) if img_match else None

        # Try to find friend/follower counts in embedded JSON
        fc_match = re.search(r'"follower_count":\s*(\d+)', html)
        fr_match = re.search(r'"friend_count":\s*(\d+)', html)
        following = 0
        if fc_match and int(fc_match.group(1)) > followers:
            followers = int(fc_match.group(1))
        if fr_match:
            following = int(fr_match.group(1))

        # Check if it's actually a profile page
        if not full_name:
            return None

        # Skip generic pages (login, error)
        if "log in" in full_name.lower() or "facebook" == full_name.lower():
            return None

        logger.info(
            "%s: @%s — %s | followers=%d",
            source, username, full_name, followers,
        )

        return RawSocialProfile(
            platform=Platform.FACEBOOK,
            username=username,
            full_name=full_name,
            biography=bio,
            profile_pic_url=profile_pic,
            followers_count=followers,
            following_count=following,
            is_private=False,
            recent_posts=[],
        )

    @staticmethod
    def _parse_count(text: str) -> int:
        text = text.strip().replace(",", "").replace("\u00a0", "")
        try:
            return int(text)
        except ValueError:
            return 0
