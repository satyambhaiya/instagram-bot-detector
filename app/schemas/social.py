"""
Universal social media profile schemas.
Used by ALL platform providers (Instagram, Twitter/X, Facebook).
These are internal data models consumed by the feature extractor — NOT the public API.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel


class Platform(str, Enum):
    """Supported social media platforms."""
    INSTAGRAM = "instagram"
    TWITTER = "twitter"
    FACEBOOK = "facebook"


class PostData(BaseModel):
    """Data for a single social media post/tweet/snap."""
    likes_count: int = 0
    comments_count: int = 0
    timestamp: Optional[datetime] = None
    caption: str = ""
    is_video: bool = False
    media_type: str = "Image"   # "Image" | "Video" | "Sidecar" | "Tweet" | "Reel" | "Story"


class RawSocialProfile(BaseModel):
    """
    Normalized social media profile data returned by any provider.
    Universal schema that maps across Instagram, Twitter/X, Facebook.

    Platform-specific mapping:
      - Twitter: followers → followers_count, friends → following_count,
                 tweets → posts_count, tweet text → caption
      - Facebook: friends → following_count + followers_count,
                  posts/reactions → likes_count
    """
    platform: Platform = Platform.INSTAGRAM
    username: str
    full_name: str = ""
    biography: str = ""
    profile_pic_url: Optional[str] = None
    followers_count: int = 0
    following_count: int = 0
    posts_count: int = 0
    is_verified: bool = False
    is_private: bool = False
    is_business: bool = False
    external_url: Optional[str] = None

    # Recent posts — providers typically return the latest 12
    recent_posts: list[PostData] = []

    # Some providers expose this directly; otherwise we estimate it
    account_created_at: Optional[datetime] = None


class ProfileSummary(BaseModel):
    """
    Trimmed public profile info returned to the frontend in the API response.
    Contains only what is safe and useful to display in the UI.
    """
    platform: str
    username: str
    display_name: str
    bio: str
    profile_pic_url: Optional[str]
    followers_count: int
    following_count: int
    posts_count: int
    is_verified: bool
    is_private: bool
    is_business: bool
    external_url: Optional[str]
