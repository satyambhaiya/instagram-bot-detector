"""
Backward-compatible re-exports.
All schemas have moved to app.schemas.social for multi-platform support.
"""

from app.schemas.social import (  # noqa: F401
    PostData,
    RawSocialProfile as RawInstagramProfile,
    ProfileSummary as InstagramProfileSummary,
)
