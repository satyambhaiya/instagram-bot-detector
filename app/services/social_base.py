"""
Abstract base class for ALL social media data providers.
Every platform provider (Instagram, Twitter, Facebook, Snapchat) implements this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.schemas.social import RawSocialProfile


class SocialMediaProvider(ABC):
    """Fetch public social media profile data for a given username."""

    @abstractmethod
    async def get_profile(self, username: str) -> RawSocialProfile:
        """
        Retrieve profile + recent posts for `username`.

        Raises:
            InstagramUserNotFoundError  — account does not exist (reused for all platforms)
            InstagramPrivateAccountError — account is private (partial data)
            InstagramProviderError      — upstream API error
            RateLimitError              — rate limit hit
        """
        ...
