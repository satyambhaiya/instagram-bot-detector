"""
Application settings — loaded from environment variables / .env file.
Access anywhere via:  from app.config import settings
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ───────────────────────────────────────────────────────────────────
    debug: bool = False

    # CORS: comma-separated string in .env → parsed into a list
    cors_origins: str = "*"

    @property
    def cors_origins_list(self) -> list[str]:
        if self.cors_origins == "*":
            return ["*"]
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    # ── Platform providers ─────────────────────────────────────────────────────
    # Each platform can be set to "mock" (dev/testing) or "http" (real scraping)
    instagram_provider: Literal["mock", "http", "instaloader", "apify", "rapidapi"] = "mock"
    twitter_provider: Literal["mock", "http", "apify"] = "mock"
    facebook_provider: Literal["mock", "http", "apify"] = "mock"
    # Instagram-specific keys
    apify_api_token: str = ""
    apify_actor_id: str = "apify~instagram-profile-scraper"
    rapidapi_key: str = ""
    rapidapi_host: str = "instagram-scraper-api2.p.rapidapi.com"

    # ── Model artifacts directory ─────────────────────────────────────────────
    model_artifacts_dir: Path = (
        Path(__file__).resolve().parent.parent / "model" / "artifacts"
    )

    # ── Cache & history ───────────────────────────────────────────────────────
    cache_ttl_seconds: int = 3600
    history_max_size: int = 500


settings = Settings()
