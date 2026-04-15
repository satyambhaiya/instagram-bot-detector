"""
BotScan API — Entry Point
=========================
Multi-platform social media bot detection API.
Supports: Instagram, Twitter/X, Facebook.

Start the server:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

API docs:
    http://localhost:8000/docs       (Swagger UI)
    http://localhost:8000/redoc      (ReDoc)
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.v1.router import router as v1_router
from app.config import settings
from app.core.exceptions import BotScanException
from app.core.logging import get_logger, setup_logging
from app.schemas.social import Platform, RawSocialProfile
from app.services.history import HistoryService
from app.services.predictor import Predictor
from app.services.social_base import SocialMediaProvider

setup_logging(debug=settings.debug)
logger = get_logger(__name__)


# ── Lifespan: load heavy resources once at startup ────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting BotScan API...")

    # Load ML model
    app.state.predictor = Predictor(artifacts_dir=settings.model_artifacts_dir)

    # Build providers for ALL platforms
    app.state.providers: dict[Platform, SocialMediaProvider] = _build_all_providers()

    # Shared history + cache service
    app.state.history_service = HistoryService(
        max_size=settings.history_max_size,
        cache_ttl_seconds=settings.cache_ttl_seconds,
    )

    platforms = [p.value for p in app.state.providers.keys()]
    logger.info(
        "Ready — platforms=%s | device=%s",
        platforms,
        app.state.predictor.device,
    )
    yield
    logger.info("Shutting down BotScan API.")


def _build_all_providers() -> dict[Platform, SocialMediaProvider]:
    """Build a provider instance for each supported platform."""
    providers: dict[Platform, SocialMediaProvider] = {}

    # ── Instagram ──────────────────────────────────────────────────────────
    providers[Platform.INSTAGRAM] = _build_instagram_provider()

    # ── Twitter/X ──────────────────────────────────────────────────────────
    providers[Platform.TWITTER] = _build_twitter_provider()

    # ── Facebook ───────────────────────────────────────────────────────────
    providers[Platform.FACEBOOK] = _build_facebook_provider()

    return providers


def _build_instagram_provider() -> SocialMediaProvider:
    """Factory: selects the Instagram provider from config."""
    provider = settings.instagram_provider

    if provider == "http":
        from app.services.instagram.http_scraper import HttpScraperProvider
        return HttpScraperProvider()

    if provider == "instaloader":
        from app.services.instagram.instaloader_provider import InstaLoaderProvider
        return InstaLoaderProvider()

    if provider == "apify":
        from app.services.instagram.apify import ApifyProvider
        return ApifyProvider(
            api_token=settings.apify_api_token,
            actor_id=settings.apify_actor_id,
        )

    if provider == "rapidapi":
        from app.services.instagram.rapidapi import RapidAPIProvider
        return RapidAPIProvider(
            api_key=settings.rapidapi_key,
            host=settings.rapidapi_host,
        )

    from app.services.instagram.mock import MockProvider
    return MockProvider()


def _build_twitter_provider() -> SocialMediaProvider:
    """Factory: selects the Twitter provider from config."""
    provider = settings.twitter_provider

    if provider == "apify":
        from app.services.twitter.apify import TwitterApifyProvider
        from app.services.twitter.http_scraper import TwitterHttpScraper
        return _TwitterApifyWithFallback(
            primary=TwitterApifyProvider(api_token=settings.apify_api_token),
            fallback=TwitterHttpScraper(),
        )

    if provider == "http":
        from app.services.twitter.http_scraper import TwitterHttpScraper
        return TwitterHttpScraper()

    from app.services.twitter.mock import TwitterMockProvider
    return TwitterMockProvider()


class _TwitterApifyWithFallback(SocialMediaProvider):
    """HTTP scraper first (free), Apify only when HTTP has no tweets.

    This saves Apify quota: the free HTTP scraper handles profile metadata
    and catches "not found" errors without burning a paid API call.
    Apify is only used when the account exists but HTTP couldn't get tweets.
    """

    def __init__(self, primary: SocialMediaProvider, fallback: SocialMediaProvider):
        self._apify = primary
        self._http = fallback

    async def get_profile(self, username: str) -> RawSocialProfile:
        from app.core.exceptions import InstagramProviderError, InstagramUserNotFoundError

        # Step 1: Try free HTTP scraper first (profile metadata + existence check)
        try:
            http_profile = await self._http.get_profile(username)
        except (InstagramUserNotFoundError, InstagramProviderError):
            # Account doesn't exist — no point calling Apify
            raise

        # Step 2: HTTP succeeded. If it got tweets, we're done (no Apify needed)
        if http_profile.recent_posts:
            logger.info(
                "Twitter HTTP scraper got %d posts for @%s — skipping Apify",
                len(http_profile.recent_posts), username,
            )
            return http_profile

        # Step 3: HTTP got profile but no tweets — try Apify for full tweet data
        try:
            apify_profile = await self._apify.get_profile(username)
            # Merge: keep Apify tweets + HTTP metadata (Apify may miss some fields)
            if not apify_profile.biography and http_profile.biography:
                apify_profile.biography = http_profile.biography
            if not apify_profile.profile_pic_url and http_profile.profile_pic_url:
                apify_profile.profile_pic_url = http_profile.profile_pic_url
            logger.info(
                "Twitter Apify enriched @%s with %d tweets",
                username, len(apify_profile.recent_posts),
            )
            return apify_profile
        except (InstagramUserNotFoundError, InstagramProviderError) as exc:
            logger.info(
                "Apify enrichment failed for @%s (%s) — using HTTP-only profile",
                username, exc,
            )
            return http_profile


def _build_facebook_provider() -> SocialMediaProvider:
    """Factory: selects the Facebook provider from config."""
    provider = settings.facebook_provider

    if provider == "apify":
        from app.services.facebook.apify import FacebookApifyProvider
        from app.services.facebook.http_scraper import FacebookHttpScraper
        return _FacebookApifyWithFallback(
            primary=FacebookApifyProvider(api_token=settings.apify_api_token),
            fallback=FacebookHttpScraper(),
        )

    if provider == "http":
        from app.services.facebook.http_scraper import FacebookHttpScraper
        return FacebookHttpScraper()

    from app.services.facebook.mock import FacebookMockProvider
    return FacebookMockProvider()


class _FacebookApifyWithFallback(SocialMediaProvider):
    """HTTP scraper first (free), Apify only to enrich with posts.

    Facebook HTTP scraper gets profile metadata (followers, bio).
    Apify gets posts + reactions but no follower counts.
    This provider: HTTP first (free existence check + metadata),
    then Apify only if we need post data.
    """

    def __init__(self, primary: SocialMediaProvider, fallback: SocialMediaProvider):
        self._apify = primary
        self._http = fallback

    async def get_profile(self, username: str) -> RawSocialProfile:
        from app.core.exceptions import InstagramProviderError, InstagramUserNotFoundError

        # Step 1: Try free HTTP scraper first (profile metadata + existence check)
        http_profile = None
        try:
            http_profile = await self._http.get_profile(username)
        except (InstagramUserNotFoundError, InstagramProviderError):
            # Account doesn't exist — no point calling Apify
            raise
        except Exception as exc:
            logger.warning("Facebook HTTP scraper failed for @%s: %s", username, exc)

        # Step 2: Try Apify for post data (reactions, comments, timestamps)
        try:
            apify_profile = await self._apify.get_profile(username)
            # Merge: Apify posts + HTTP metadata
            if http_profile:
                apify_profile.followers_count = http_profile.followers_count or apify_profile.followers_count
                apify_profile.following_count = http_profile.following_count or apify_profile.following_count
                apify_profile.biography = http_profile.biography or apify_profile.biography
                apify_profile.full_name = http_profile.full_name or apify_profile.full_name
                apify_profile.profile_pic_url = http_profile.profile_pic_url or apify_profile.profile_pic_url
                apify_profile.posts_count = max(apify_profile.posts_count, http_profile.posts_count)
                logger.info(
                    "Facebook merged: Apify posts (%d) + HTTP metadata (followers=%d)",
                    len(apify_profile.recent_posts), apify_profile.followers_count,
                )
            return apify_profile
        except (InstagramUserNotFoundError, InstagramProviderError) as exc:
            logger.info("Facebook Apify failed for @%s (%s)", username, exc)
            if http_profile:
                logger.info("Using HTTP-only profile for @%s", username)
                return http_profile
            raise


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="BotScan API",
    description=(
        "AI-powered social media account classifier. "
        "Supports Instagram, Twitter/X, and Facebook. "
        "Submit a username + platform → get a Human / Bot / Suspicious verdict."
    ),
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount versioned API router
app.include_router(v1_router)


# ── Global exception handler ─────────────────────────────────────────────────

@app.exception_handler(BotScanException)
async def botscan_exception_handler(request: Request, exc: BotScanException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


# ── Frontend ─────────────────────────────────────────────────────────────────

_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
_INDEX_HTML = _FRONTEND_DIR / "botscan.html"


@app.get("/", include_in_schema=False)
async def root():
    """Serve the BotScan frontend."""
    if _INDEX_HTML.exists():
        return HTMLResponse(_INDEX_HTML.read_text(encoding="utf-8"))
    return {
        "service": "BotScan API",
        "version": "2.0.0",
        "platforms": ["instagram", "twitter", "facebook"],
        "docs": "/docs",
        "health": "/api/v1/health",
    }
