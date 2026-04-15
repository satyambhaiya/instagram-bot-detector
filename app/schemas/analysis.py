"""
Public API schemas — request and response models for the /analyze endpoint.
These are the contracts exposed to the frontend developer.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

from app.schemas.social import Platform, ProfileSummary


# ── Request ───────────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    username: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Username or handle (without @).",
        examples=["natgeo", "cristiano", "elonmusk"],
    )
    platform: Platform = Field(
        Platform.INSTAGRAM,
        description="Social media platform to analyze.",
    )

    @field_validator("username")
    @classmethod
    def clean_username(cls, v: str) -> str:
        v = v.strip().lstrip("@").lower()
        if not re.match(r"^[a-z0-9._]{1,50}$", v):
            raise ValueError(
                "Username may only contain letters, numbers, dots, and underscores."
            )
        return v


# ── Prediction sub-models ─────────────────────────────────────────────────────

class AnalysisProbabilities(BaseModel):
    human: float = Field(..., ge=0, le=1, description="Probability of being a human account.")
    bot: float = Field(..., ge=0, le=1, description="Probability of being a bot account.")
    suspicious: float = Field(..., ge=0, le=1, description="Probability of being a suspicious account.")


class AnalysisResult(BaseModel):
    prediction: Literal["Human", "Bot", "Suspicious"]
    confidence: float = Field(..., ge=0, le=1, description="Confidence for the top prediction.")
    risk_score: float = Field(
        ..., ge=0, le=1,
        description="Composite risk score: P(Bot) + 0.5 × P(Suspicious). Higher = more risk.",
    )
    risk_level: Literal["Low", "Medium", "High", "Critical"] = Field(
        ..., description="Low (<0.25) | Medium (<0.50) | High (<0.75) | Critical (≥0.75)."
    )
    probabilities: AnalysisProbabilities


class BehavioralAnalysis(BaseModel):
    """Human-readable behavioral breakdown shown on the frontend."""
    activity_pattern: Literal["Natural", "Moderately Irregular", "Highly Automated"] = Field(
        ..., description="Overall posting behavior pattern.",
    )
    engagement_rate: Literal["High", "Moderate", "Low", "Inflated / Artificial"] = Field(
        ..., description="Quality and level of engagement.",
    )
    posting_frequency: Literal["Consistent", "Irregular", "Excessive", "Minimal"] = Field(
        ..., description="How often the account posts.",
    )
    account_age: Literal["New", "Growing", "Established", "Old"] = Field(
        ..., description="Account maturity based on age.",
    )


class DataQuality(BaseModel):
    """Describes how reliable the feature set was for this analysis."""
    completeness: float = Field(
        ..., ge=0, le=1,
        description="Fraction of features directly observed vs estimated (1.0 = fully observed).",
    )
    is_private_account: bool = Field(
        False,
        description="True if the account is private (reduced data quality).",
    )
    estimated_features: list[str] = Field(
        default_factory=list,
        description="Features that could not be directly observed and were statistically estimated.",
    )
    note: Optional[str] = Field(
        None,
        description="Human-readable note about data quality, e.g. for private accounts.",
    )


# ── Main response ─────────────────────────────────────────────────────────────

class AnalyzeResponse(BaseModel):
    username: str
    platform: str = "instagram"
    profile: ProfileSummary
    analysis: AnalysisResult
    behavioral_analysis: BehavioralAnalysis = Field(
        ..., description="Human-readable behavioral breakdown.",
    )
    features: dict = Field(
        ...,
        description="All 27 ML features extracted from the account (before scaling).",
    )
    data_quality: DataQuality
    analyzed_at: datetime
    cached: bool = Field(False, description="True if this result was served from cache.")


# ── History / Stats ───────────────────────────────────────────────────────────

class HistoryEntry(BaseModel):
    username: str
    platform: str = "instagram"
    prediction: Literal["Human", "Bot", "Suspicious"]
    risk_score: float
    risk_level: Literal["Low", "Medium", "High", "Critical"]
    confidence: float
    analyzed_at: datetime
    cached: bool


class HistoryResponse(BaseModel):
    total: int
    results: list[HistoryEntry]


class StatsResponse(BaseModel):
    total_analyses: int
    distribution: dict[str, float] = Field(
        ...,
        description="Fraction of each prediction class over all analyses.",
    )
    avg_risk_score: float
    high_risk_rate: float = Field(
        ...,
        description="Fraction of analyses with risk_score >= 0.65.",
    )
