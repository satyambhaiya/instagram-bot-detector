"""
Instagram provider — inherits from the universal SocialMediaProvider.
Kept for backward compatibility; all Instagram providers extend this.
"""

from __future__ import annotations

from app.services.social_base import SocialMediaProvider


# Alias so existing code (HttpScraperProvider, MockProvider, etc.) still works
InstagramProvider = SocialMediaProvider
