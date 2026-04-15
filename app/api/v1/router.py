"""Aggregates all v1 endpoint routers under the /api/v1 prefix."""

from fastapi import APIRouter

from app.api.v1.endpoints import analysis, health, history

router = APIRouter(prefix="/api/v1")
router.include_router(health.router)
router.include_router(analysis.router)
router.include_router(history.router)
