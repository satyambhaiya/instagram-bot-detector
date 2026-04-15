"""
Snapchat HTTP Scraper — fetches public profile data.

Strategy:
  1. snapchat.com/add/{username} — public profile page
  2. story.snapchat.com/s/{username} — public stories page

Note: Snapchat has the most limited public data of all platforms.
Most profile data is private by design. This scraper extracts
whatever is publicly visible (display name, bitmoji, public status).
Bot detection relies heavily on username patterns and profile metadata
rather than engagement metrics.
"""

from __future__ import annotations

import re
from datetime import timezone
from typing import Optional

import httpx

from app.core.exceptions import (
    InstagramProviderError,
    InstagramUserNotFoundError,
)
from app.core.logging import get_logger
from app.schemas.social import Platform, PostData, RawSocialProfile
from app.services.social_base import SocialMediaProvider

logger = get_logger(__name__)
_UTC = timezone.utc
_TIMEOUT = 15.0

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


class SnapchatHttpScraper(SocialMediaProvider):
    """Scrape public Snapchat profile data."""

    async def get_profile(self, username: str) -> RawSocialProfile:
        username = username.lstrip("@").lower()

        # Try the public add page
        profile = await self._try_add_page(username)
        if profile:
            return profile

        # Try the story page
        profile = await self._try_story_page(username)
        if profile:
            return profile

        raise InstagramUserNotFoundError(
            f"Could not fetch Snapchat profile for '{username}'. "
            "The account may not exist or is not publicly visible."
        )

    async def _try_add_page(self, username: str) -> Optional[RawSocialProfile]:
        """Try snapchat.com/add/{username} — the public profile/add page."""
        url = f"https://www.snapchat.com/add/{username}"
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True) as client:
                resp = await client.get(url, headers=_HEADERS)

                if resp.status_code == 404:
                    return None
                if resp.status_code != 200:
                    return None

                html = resp.text
                return self._parse_add_page(html, username)
        except Exception as e:
            logger.debug("snapchat.com/add failed for @%s: %s", username, e)
            return None

    async def _try_story_page(self, username: str) -> Optional[RawSocialProfile]:
        """Try story.snapchat.com — public stories."""
        url = f"https://story.snapchat.com/s/{username}"
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True) as client:
                resp = await client.get(url, headers=_HEADERS)

                if resp.status_code != 200:
                    return None

                html = resp.text
                return self._parse_story_page(html, username)
        except Exception as e:
            logger.debug("story.snapchat.com failed for @%s: %s", username, e)
            return None

    def _parse_add_page(self, html: str, username: str) -> Optional[RawSocialProfile]:
        """Parse the snapchat.com/add page."""
        # Extract display name from og:title or page title
        title_match = re.search(r'<meta[^>]*property="og:title"[^>]*content="([^"]*)"', html)
        full_name = ""
        if title_match:
            full_name = title_match.group(1).strip()
            full_name = self._clean_snapchat_name(full_name)

        # Profile picture (bitmoji)
        img_match = re.search(r'<meta[^>]*property="og:image"[^>]*content="([^"]*)"', html)
        profile_pic = img_match.group(1) if img_match else None

        # Description — Snapchat pages return generic localized text
        # like "Eudes is on Snapchat!" / "Eudes Snapchat पर हैं!" which
        # is NOT a real user bio. Detect and discard these.
        desc_match = re.search(r'<meta[^>]*property="og:description"[^>]*content="([^"]*)"', html)
        bio = ""
        if desc_match:
            raw_desc = desc_match.group(1).strip()
            if not self._is_generic_snapchat_bio(raw_desc):
                bio = raw_desc

        # Check for "not found" indicators
        if "page not found" in html.lower() or "this page isn" in html.lower():
            return None

        # Check if we got meaningful data
        has_pic = profile_pic is not None and "bitmoji" in (profile_pic or "").lower()

        if not full_name and not has_pic:
            return None

        return RawSocialProfile(
            platform=Platform.SNAPCHAT,
            username=username,
            full_name=full_name or username,
            biography=bio,
            profile_pic_url=profile_pic,
            followers_count=0,  # Not publicly available
            following_count=0,  # Not publicly available
            posts_count=0,
            is_private=False,
            is_verified=False,
            recent_posts=[],  # Stories are ephemeral
        )

    # ── Name / bio cleaning ──────────────────────────────────────────────────

    @staticmethod
    def _clean_snapchat_name(name: str) -> str:
        """Remove localized 'on Snapchat' / 'Snapchat पर' wrappers.

        Snapchat og:title returns the display name wrapped in a localized
        phrase. Known patterns:
          English : "Name on Snapchat"
          French  : "Name sur Snapchat"
          Hindi   : "Snapchat पर Name"
          Spanish : "Name en Snapchat"
          German  : "Name auf Snapchat"
          Generic : "Snapchat - Name"  /  "Name | Snapchat"
        """
        # Suffix patterns: "Name {preposition} Snapchat"
        name = re.sub(
            r'\s*(?:on|sur|en|auf|no|на)\s+Snapchat\s*$',
            '', name, flags=re.IGNORECASE,
        )
        # Prefix patterns: "Snapchat {preposition} Name"  (e.g. Hindi पर)
        name = re.sub(
            r'^Snapchat\s*(?:पर|에서|で|上的|di|на)?\s+',
            '', name, flags=re.IGNORECASE,
        )
        # Separator patterns: "Snapchat - Name" / "Name | Snapchat"
        name = re.sub(r'\s*[|–—\-]\s*Snapchat\s*$', '', name, flags=re.IGNORECASE)
        name = re.sub(r'^Snapchat\s*[|–—\-]\s*', '', name, flags=re.IGNORECASE)
        return name.strip()

    @staticmethod
    def _is_generic_snapchat_bio(text: str) -> bool:
        """Return True if `text` is a generic Snapchat page description.

        These are auto-generated by Snapchat in every locale:
          "Eudes is on Snapchat!"
          "Eudes Snapchat पर हैं!"
          "Add Eudes on Snapchat!"
        They carry zero signal about the user, so we discard them.
        """
        t = text.lower()
        # English / French / Spanish / etc.
        if re.search(r'(?:is|est|está|ist)\s+on\s+snapchat', t):
            return True
        if re.search(r'on\s+snapchat\s*!?\s*$', t):
            return True
        if re.search(r'add\s+.*\s+on\s+snapchat', t):
            return True
        if re.search(r'sur\s+snapchat\s*!?\s*$', t):
            return True
        # Hindi "Snapchat पर हैं"
        if 'snapchat' in t and ('पर' in text or 'हैं' in text):
            return True
        # Very short generic (under 50 chars with "snapchat" in it)
        if len(text) < 50 and 'snapchat' in t:
            return True
        return False

    def _parse_story_page(self, html: str, username: str) -> Optional[RawSocialProfile]:
        """Parse the story.snapchat.com page."""
        title_match = re.search(r'<meta[^>]*property="og:title"[^>]*content="([^"]*)"', html)
        full_name = ""
        if title_match:
            full_name = self._clean_snapchat_name(title_match.group(1).strip())

        img_match = re.search(r'<meta[^>]*property="og:image"[^>]*content="([^"]*)"', html)
        profile_pic = img_match.group(1) if img_match else None

        if not full_name:
            return None

        return RawSocialProfile(
            platform=Platform.SNAPCHAT,
            username=username,
            full_name=full_name,
            profile_pic_url=profile_pic,
            recent_posts=[],
        )
