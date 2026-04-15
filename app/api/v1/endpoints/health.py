"""
Health endpoints
----------------
GET /api/v1/health        — basic liveness check
GET /api/v1/health/model  — model metadata + performance metrics
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.dependencies import get_predictor
from app.services.predictor import Predictor

router = APIRouter(tags=["Health"])


@router.get("/health", summary="Liveness check")
async def health():
    """Returns 200 OK when the API is running."""
    return {"status": "ok", "service": "BotScan API"}


@router.get("/health/model", summary="Model status and metadata")
async def model_info(predictor: Predictor = Depends(get_predictor)):
    """Returns model architecture details and test-set performance metrics."""
    meta = predictor.metadata
    return {
        "status": "ready" if predictor.is_ready else "not_ready",
        "device": predictor.device,
        "architecture": {
            "input_dim": meta.get("input_dim"),
            "hidden_dim": meta.get("hidden"),
            "dropout": meta.get("dropout"),
            "num_classes": 3,
            "classes": meta.get("label_names"),
        },
        "test_metrics": meta.get("test_metrics", {}),
        "features": meta.get("feature_cols", []),
    }
