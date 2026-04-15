"""
Analysis endpoints — the core of the BotScan API.

POST /api/v1/analyze
    Analyze a social media account on any supported platform.
    Input:  { "username": "someuser", "platform": "instagram" }
    Query:  ?force_refresh=true  — bypass cache and re-fetch
    Output: Full AnalyzeResponse (prediction, profile, features, data quality)

GET  /api/v1/analyze/{platform}/{username}
    Return the cached result for a platform+username if available.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.core.exceptions import (
    BotScanException,
    InstagramPrivateAccountError,
    InstagramUserNotFoundError,
)
from app.core.logging import get_logger
from app.dependencies import get_history_service, get_predictor, get_provider
from app.schemas.analysis import (
    AnalysisResult,
    AnalysisProbabilities,
    AnalyzeRequest,
    AnalyzeResponse,
    BehavioralAnalysis,
    DataQuality,
)
from app.schemas.social import Platform, ProfileSummary, RawSocialProfile
from app.services.feature_extractor import FEATURE_COLS, extract_features
from app.services.history import HistoryService
from app.services.predictor import PredictionOutput, Predictor
from app.services.social_base import SocialMediaProvider

logger = get_logger(__name__)
router = APIRouter(tags=["Analysis"])

_UTC = timezone.utc

# Only cache results that meet these quality thresholds
_MIN_COMPLETENESS_TO_CACHE = 0.75
_MIN_CONFIDENCE_TO_CACHE = 0.60


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze a social media account",
    description=(
        "Fetches account data from the specified platform, extracts 27 behavioral "
        "features, and classifies it as **Human**, **Bot**, or **Suspicious**. "
        "Supports: instagram, twitter, facebook."
    ),
)
async def analyze(
    request: AnalyzeRequest,
    force_refresh: bool = Query(
        False,
        description="Bypass the cache and re-fetch data from the platform.",
    ),
    predictor: Predictor = Depends(get_predictor),
    history: HistoryService = Depends(get_history_service),
) -> AnalyzeResponse:
    username = request.username
    platform = request.platform

    # Resolve the provider for this platform via app.state
    from fastapi import Request as _Req
    # We use get_provider manually since platform comes from the body, not a path param
    # This is handled below via the _get_provider_from_app helper

    # ── 1. Check cache (skip if force_refresh) ─────────────────────────────────
    cache_key = f"{platform.value}:{username}"
    if not force_refresh:
        cached = history.get_cached(cache_key)
        if cached:
            logger.info("Cache hit for @%s on %s", username, platform.value)
            return cached.model_copy(update={"cached": True})

    # ── 2. Fetch profile data (with retry for incomplete scrapes) ──────────────
    logger.info("Analyzing @%s on %s (force_refresh=%s)", username, platform.value, force_refresh)
    provider = _get_provider_from_context(platform)
    profile, private_mode = await _fetch_profile(provider, username, platform)

    # Retry once if the scrape returned no posts for a public, non-empty account
    if (
        not private_mode
        and not profile.recent_posts
        and profile.posts_count > 0
    ):
        logger.warning(
            "@%s on %s: got 0 posts on first attempt (expected %d) — retrying",
            username, platform.value, profile.posts_count,
        )
        await asyncio.sleep(1.5)
        profile, private_mode = await _fetch_profile(provider, username, platform)

    # ── 3. Extract features ───────────────────────────────────────────────────
    extraction = extract_features(profile)
    features = extraction.features
    estimated_features = extraction.estimated

    # ── 4. Compute data completeness ──────────────────────────────────────────
    directly_observed = len(FEATURE_COLS) - len(estimated_features)
    completeness = round(directly_observed / len(FEATURE_COLS), 3)

    # ── 5. Run model inference (with data completeness penalty) ───────────────
    prediction_output = predictor.predict(features, data_completeness=completeness)

    # ── 5b. Heuristic override for obvious bot/spam patterns ────────────────
    prediction_output = _apply_bot_heuristics(prediction_output, features, profile)

    # ── 6. Build response ─────────────────────────────────────────────────────
    quality_note = _build_quality_note(private_mode, completeness, profile)

    response = AnalyzeResponse(
        username=username,
        platform=platform.value,
        profile=_build_profile_summary(profile),
        analysis=AnalysisResult(
            prediction=prediction_output.prediction,
            confidence=prediction_output.confidence,
            risk_score=prediction_output.risk_score,
            risk_level=prediction_output.risk_level,
            probabilities=AnalysisProbabilities(
                human=prediction_output.probabilities.get("Human", 0.0),
                bot=prediction_output.probabilities.get("Bot", 0.0),
                suspicious=prediction_output.probabilities.get("Suspicious", 0.0),
            ),
        ),
        behavioral_analysis=_compute_behavioral_analysis(features, prediction_output.prediction),
        features=features,
        data_quality=DataQuality(
            completeness=completeness,
            is_private_account=private_mode,
            estimated_features=estimated_features,
            note=quality_note,
        ),
        analyzed_at=datetime.now(_UTC),
        cached=False,
    )

    # ── 7. Selective caching — only store high-quality results ─────────────────
    if (
        completeness >= _MIN_COMPLETENESS_TO_CACHE
        and prediction_output.confidence >= _MIN_CONFIDENCE_TO_CACHE
    ):
        history.set_cache(cache_key, response)
        logger.info(
            "@%s/%s → %s (risk=%.3f, confidence=%.3f) [cached]",
            platform.value, username,
            prediction_output.prediction,
            prediction_output.risk_score,
            prediction_output.confidence,
        )
    else:
        logger.info(
            "@%s/%s → %s (risk=%.3f, confidence=%.3f) [NOT cached: completeness=%.2f]",
            platform.value, username,
            prediction_output.prediction,
            prediction_output.risk_score,
            prediction_output.confidence,
            completeness,
        )

    history.add(response)
    return response


@router.get(
    "/analyze/{platform}/{username}",
    response_model=AnalyzeResponse,
    summary="Get cached analysis for a platform and username",
    description="Returns the cached analysis result if available (HTTP 200), or 404 if not cached.",
)
async def get_cached_analysis(
    platform: str,
    username: str,
    history: HistoryService = Depends(get_history_service),
) -> AnalyzeResponse:
    cache_key = f"{platform}:{username.lower()}"
    cached = history.get_cached(cache_key)
    if not cached:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No cached analysis for '@{username}' on {platform}. Use POST /analyze.",
        )
    return cached.model_copy(update={"cached": True})


# Keep old GET endpoint for backward compat
@router.get(
    "/analyze/{username}",
    response_model=AnalyzeResponse,
    summary="Get cached analysis (Instagram, backward compat)",
    include_in_schema=False,
)
async def get_cached_analysis_legacy(
    username: str,
    history: HistoryService = Depends(get_history_service),
) -> AnalyzeResponse:
    # Try instagram first, then any platform
    for prefix in ["instagram", "twitter", "facebook"]:
        cache_key = f"{prefix}:{username.lower()}"
        cached = history.get_cached(cache_key)
        if cached:
            return cached.model_copy(update={"cached": True})
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"No cached analysis found for '@{username}'.",
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

# Thread-local storage for the current request's app instance
import contextvars
_current_app = contextvars.ContextVar("_current_app")


def _get_provider_from_context(platform: Platform) -> SocialMediaProvider:
    """Get provider from the app state. Works via middleware context."""
    from app.main import app
    providers = app.state.providers
    provider = providers.get(platform)
    if not provider:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Platform '{platform.value}' is not supported.",
        )
    return provider


async def _fetch_profile(
    provider: SocialMediaProvider, username: str, platform: Platform
) -> tuple[RawSocialProfile, bool]:
    """Fetch a social media profile with automatic retry on transient failures."""
    max_attempts = 2
    last_exc = None

    for attempt in range(1, max_attempts + 1):
        private_mode = False
        try:
            profile = await provider.get_profile(username)
            return profile, private_mode
        except InstagramUserNotFoundError:
            # Don't retry "not found" — the account doesn't exist on this
            # platform, retrying just wastes Apify quota.
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Account '@{username}' was not found on {platform.value}.",
            )
        except InstagramPrivateAccountError:
            logger.warning("@%s on %s is private — limited analysis", username, platform.value)
            private_mode = True
            profile = RawSocialProfile(
                platform=platform,
                username=username,
                is_private=True,
                recent_posts=[],
            )
            return profile, private_mode
        except BotScanException as exc:
            if attempt < max_attempts:
                logger.warning(
                    "@%s on %s: provider error on attempt %d — retrying: %s",
                    username, platform.value, attempt, exc.detail,
                )
                await asyncio.sleep(2.0)
                last_exc = exc
                continue
            raise HTTPException(status_code=exc.status_code, detail=exc.detail)
        except Exception as exc:
            if attempt < max_attempts:
                logger.warning(
                    "@%s on %s: transient error on attempt %d — retrying: %s",
                    username, platform.value, attempt, exc,
                )
                await asyncio.sleep(2.0)
                last_exc = exc
                continue
            logger.exception("Unexpected error fetching @%s on %s: %s", username, platform.value, exc)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"{platform.value} data provider is temporarily unavailable.",
            )

    # Should never reach here, but just in case
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=f"{platform.value} data provider is temporarily unavailable.",
    )


def _build_quality_note(
    private_mode: bool, completeness: float, profile: RawSocialProfile
) -> str | None:
    if private_mode:
        return (
            "Account is private. Post-level features could not be retrieved; "
            "the analysis is based on public profile data only and may be less accurate."
        )
    if completeness < _MIN_COMPLETENESS_TO_CACHE:
        platform_name = profile.platform.value.capitalize()
        n_posts = len(profile.recent_posts)
        has_activity = profile.posts_count > 0

        if n_posts == 0 and has_activity:
            return (
                f"{platform_name} does not expose individual post data publicly. "
                f"Profile shows {profile.posts_count:,} posts, {profile.followers_count:,} followers, "
                f"and {profile.following_count:,} following. "
                "Engagement metrics were estimated from these profile statistics. "
                f"Data completeness: {completeness:.0%}."
            )
        if n_posts == 0:
            return (
                f"No post activity found for @{profile.username} on {platform_name}. "
                "The analysis relies entirely on profile metadata and may be less accurate. "
                f"Data completeness: {completeness:.0%}."
            )
        return (
            f"Only {n_posts} of {profile.posts_count:,} posts could be analyzed on {platform_name}. "
            f"Data completeness: {completeness:.0%}."
        )
    return None


_SPAM_KEYWORDS = {"free", "followers", "follow4follow", "f4f", "promo", "hack",
                   "likes", "gainwithxchina", "gain", "shoutout", "dm4dm",
                   "bot", "spam"}


def _apply_bot_heuristics(
    pred: PredictionOutput,
    features: dict[str, float],
    profile: RawSocialProfile,
) -> PredictionOutput:
    """Override ML prediction when observable signals clearly indicate a bot.

    The ML model can be fooled when estimated features default to human-like
    values. This layer catches obvious patterns the model misses.
    Also upgrades Suspicious → Bot when signals are overwhelming.
    """
    if pred.prediction == "Bot":
        return pred  # model already flagged it as Bot — trust it

    bot_signals = 0

    # ── Behavioral signals (require COMBINATIONS, not single indicators) ───
    ppd = features.get("posts_per_day", 0)
    night = features.get("night_activity_ratio", 0)
    following = features.get("following_count", 0)
    followers = features.get("followers_count", 0)
    hashtags = features.get("hashtags_per_post", 0)
    mentions = features.get("mentions_per_post", 0)
    ftf_ratio = features.get("following_to_followers_ratio", 0)
    lcr = features.get("likes_comments_ratio", 0)
    regularity = features.get("posting_regularity", 0.5)

    # High volume + round-the-clock → automation
    if ppd > 5 and night > 0.40:
        bot_signals += 2

    # High volume + follows nobody → broadcast bot
    if ppd > 5 and following == 0 and followers > 50:
        bot_signals += 2

    # Nocturnal posting pattern (exclude large accounts — US daytime = UTC night)
    if night > 0.45 and ppd > 1 and followers < 500_000:
        bot_signals += 1

    # Broadcast-only (follows nobody)
    if following == 0 and followers > 100 and followers < 500_000:
        bot_signals += 1

    # Zero content diversity at high volume
    if hashtags == 0 and mentions == 0 and ppd > 5:
        bot_signals += 1

    # ── Mass-follow pattern (Sukhali: "too many following and less followers") ──
    if ftf_ratio > 10 and followers < 500:
        bot_signals += 2  # follows 10x more than followers → mass-follow bot
    elif ftf_ratio > 5 and followers < 1000:
        bot_signals += 1

    # ── Engagement quality (Sukhali: "likes vs comments ratio") ──────────
    # Very high likes but almost no comments → likely bought likes
    # Exclude mega-accounts (>1M followers): high lcr is natural at that scale
    if lcr > 80 and followers > 100 and followers < 1_000_000:
        bot_signals += 1

    # ── Posting regularity (Sukhali: "fixed intervals vs random") ────────
    if regularity < 0.10 and ppd > 2:
        bot_signals += 1  # extremely regular posting + active = automation

    # Automated broadcast: follows nobody + near-zero timing variance
    # Catches bots like @year_progress that post 1x/day at exact same time
    if following == 0 and regularity < 0.05 and followers > 50:
        bot_signals += 2

    # ── Username spam patterns ─────────────────────────────────────────────
    uname = (profile.username or "").lower()
    uname_parts = set(uname.replace("_", ".").replace("-", ".").split("."))
    if uname_parts & _SPAM_KEYWORDS:
        bot_signals += 3

    # Username ends with "bot" — strong signal (earthquakebot, emojimashupbot)
    if uname.endswith("bot"):
        bot_signals += 3

    # ── Bio declares itself as a bot ───────────────────────────────────────
    bio = (profile.biography or "").lower()
    _bot_bio_patterns = [" bot ", " bot.", " bot,", " bot!", "bot made by",
                         "automated", "i am a bot", "i'm a bot",
                         "i am a robot", "i'm a robot", " robot ",
                         "parody"]
    if any(p in f" {bio} " for p in _bot_bio_patterns):
        bot_signals += 3

    # ── Empty profile + suspicious name ────────────────────────────────────
    if (features.get("bio_length", 0) == 0
            and features.get("profile_pic", 1) == 0
            and followers < 10):
        bot_signals += 1

    # ── Apply override ─────────────────────────────────────────────────────
    # Already Suspicious? Only upgrade to Bot if signals are strong
    if pred.prediction == "Suspicious":
        if bot_signals >= 3:
            logger.info("Heuristic upgrade Suspicious → Bot (signals=%d) for @%s", bot_signals, profile.username)
            return PredictionOutput(
                prediction="Bot",
                confidence=min(0.70 + bot_signals * 0.03, 0.92),
                risk_score=min(0.75 + bot_signals * 0.02, 0.95),
                risk_level="Critical" if bot_signals >= 6 else "High",
                probabilities={"Human": 0.05, "Bot": 0.85, "Suspicious": 0.10},
            )
        return pred  # keep Suspicious

    # Human prediction — override if enough bot signals
    if bot_signals >= 4:
        logger.info("Heuristic override → Bot (signals=%d) for @%s", bot_signals, profile.username)
        return PredictionOutput(
            prediction="Bot",
            confidence=min(0.70 + bot_signals * 0.03, 0.92),
            risk_score=min(0.75 + bot_signals * 0.02, 0.95),
            risk_level="Critical" if bot_signals >= 6 else "High",
            probabilities={"Human": 0.05, "Bot": 0.85, "Suspicious": 0.10},
        )
    if bot_signals >= 2:
        logger.info("Heuristic override → Suspicious (signals=%d) for @%s", bot_signals, profile.username)
        return PredictionOutput(
            prediction="Suspicious",
            confidence=min(0.60 + bot_signals * 0.05, 0.85),
            risk_score=min(0.50 + bot_signals * 0.05, 0.80),
            risk_level="High",
            probabilities={"Human": 0.25, "Bot": 0.25, "Suspicious": 0.50},
        )

    return pred


def _compute_behavioral_analysis(features: dict, prediction: str) -> BehavioralAnalysis:
    """Compute human-readable behavioral labels from raw features."""
    ppd = features.get("posts_per_day", 0)
    night = features.get("night_activity_ratio", 0)
    age = features.get("account_age_days", 0)
    following = features.get("following_count", 0)
    followers = features.get("followers_count", 0)
    avg_likes = features.get("avg_likes_per_post", 0)
    avg_comments = features.get("avg_comments_per_post", 0)
    ftf_ratio = features.get("following_to_followers_ratio", 0)
    lcr = features.get("likes_comments_ratio", 0)
    regularity = features.get("posting_regularity", 0.5)

    # ── Activity Pattern ───────────────────────────────────────────────────
    if (ppd > 5 and night > 0.40) or (ppd > 10 and following == 0):
        activity = "Highly Automated"
    elif regularity < 0.10 and ppd > 2:
        activity = "Highly Automated"   # fixed-interval posting
    elif ppd > 5 or night > 0.35 or (following == 0 and followers > 100):
        activity = "Moderately Irregular"
    elif regularity < 0.20 and ppd > 1:
        activity = "Moderately Irregular"
    else:
        activity = "Natural"

    # Override based on prediction when signals are borderline
    if prediction == "Bot" and activity == "Natural":
        activity = "Moderately Irregular"

    # ── Engagement Rate ────────────────────────────────────────────────────
    raw_er = avg_likes / max(followers, 1)
    if followers < 1000:
        if raw_er > 0.15:
            eng_label = "High"
        elif raw_er > 0.03:
            eng_label = "Moderate"
        else:
            eng_label = "Low"
    else:
        if raw_er > 0.06:
            eng_label = "High"
        elif raw_er > 0.015:
            eng_label = "Moderate"
        elif raw_er > 0.003:
            eng_label = "Low"
        else:
            eng_label = "Low"

    # Detect inflated engagement: high likes but almost no comments (bought likes)
    # Exclude mega-accounts: high lcr is natural for 1M+ followers
    if lcr > 80 and followers > 100 and followers < 1_000_000:
        eng_label = "Inflated / Artificial"
    elif avg_likes > 0 and avg_comments > 0:
        comment_ratio = avg_comments / avg_likes
        if raw_er > 0.05 and comment_ratio < 0.005 and followers > 10_000:
            eng_label = "Inflated / Artificial"

    if prediction == "Bot" and eng_label == "High":
        eng_label = "Inflated / Artificial"

    # ── Posting Frequency ──────────────────────────────────────────────────
    if ppd > 5:
        freq = "Excessive"
    elif ppd < 0.05:
        freq = "Minimal"
    elif ppd < 0.15:
        freq = "Irregular"
    else:
        freq = "Consistent"

    # ── Account Age ────────────────────────────────────────────────────────
    if age < 90:
        age_label = "New"
    elif age < 365:
        age_label = "Growing"
    elif age < 1825:  # 5 years
        age_label = "Established"
    else:
        age_label = "Old"

    return BehavioralAnalysis(
        activity_pattern=activity,
        engagement_rate=eng_label,
        posting_frequency=freq,
        account_age=age_label,
    )


def _build_profile_summary(profile: RawSocialProfile) -> ProfileSummary:
    return ProfileSummary(
        platform=profile.platform.value,
        username=profile.username,
        display_name=profile.full_name or profile.username,
        bio=profile.biography or "",
        profile_pic_url=profile.profile_pic_url,
        followers_count=profile.followers_count,
        following_count=profile.following_count,
        posts_count=profile.posts_count,
        is_verified=profile.is_verified,
        is_private=profile.is_private,
        is_business=profile.is_business,
        external_url=profile.external_url,
    )
