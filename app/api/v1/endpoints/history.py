"""
History & Stats endpoints
--------------------------
GET    /api/v1/history          — paginated analysis history
DELETE /api/v1/history          — clear history + cache
GET    /api/v1/history/export   — download history as CSV
GET    /api/v1/stats            — aggregated statistics
"""

from __future__ import annotations

import csv
import io

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse

from app.dependencies import get_history_service
from app.schemas.analysis import HistoryResponse, StatsResponse
from app.services.history import HistoryService

router = APIRouter(tags=["History & Stats"])


@router.get(
    "/history",
    response_model=HistoryResponse,
    summary="Get analysis history",
)
async def get_history(
    limit: int = Query(50, ge=1, le=500, description="Max number of entries to return."),
    history: HistoryService = Depends(get_history_service),
) -> HistoryResponse:
    entries = history.get_history(limit=limit)
    return HistoryResponse(total=len(entries), results=entries)


@router.delete(
    "/history",
    summary="Clear history and cache",
    status_code=status.HTTP_200_OK,
)
async def clear_history(
    history: HistoryService = Depends(get_history_service),
):
    history.clear()
    return {"message": "History and cache cleared."}


@router.get(
    "/history/export",
    summary="Export history as CSV",
    response_class=StreamingResponse,
)
async def export_history(
    history: HistoryService = Depends(get_history_service),
):
    entries = history.get_history(limit=500)
    if not entries:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No analysis history to export.",
        )

    fields = ["username", "platform", "prediction", "risk_score", "risk_level", "confidence", "analyzed_at", "cached"]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields)
    writer.writeheader()
    for entry in entries:
        writer.writerow({
            "username":    entry.username,
            "platform":    getattr(entry, "platform", "instagram"),
            "prediction":  entry.prediction,
            "risk_score":  entry.risk_score,
            "risk_level":  entry.risk_level,
            "confidence":  entry.confidence,
            "analyzed_at": entry.analyzed_at.isoformat(),
            "cached":      entry.cached,
        })
    buf.seek(0)

    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=botscan_history.csv"},
    )


@router.get(
    "/stats",
    response_model=StatsResponse,
    summary="Aggregated statistics over all analyses",
)
async def get_stats(
    history: HistoryService = Depends(get_history_service),
) -> StatsResponse:
    data = history.stats()
    if data["total_analyses"] == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No analyses have been run yet.",
        )
    return StatsResponse(**data)
