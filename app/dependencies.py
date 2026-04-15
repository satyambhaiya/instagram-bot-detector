"""
FastAPI dependency injection.
Singletons are stored on `app.state` and injected into route handlers via Depends().
"""

from __future__ import annotations

from fastapi import Request

from app.schemas.social import Platform
from app.services.history import HistoryService
from app.services.predictor import Predictor
from app.services.social_base import SocialMediaProvider


def get_predictor(request: Request) -> Predictor:
    return request.app.state.predictor


def get_provider(request: Request, platform: Platform) -> SocialMediaProvider:
    """Get the appropriate provider for the given platform."""
    providers: dict[Platform, SocialMediaProvider] = request.app.state.providers
    provider = providers.get(platform)
    if not provider:
        raise ValueError(f"No provider configured for platform: {platform.value}")
    return provider


def get_history_service(request: Request) -> HistoryService:
    return request.app.state.history_service
