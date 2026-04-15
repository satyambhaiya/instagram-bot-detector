"""
History & Cache Service
=======================
Two responsibilities:

1. **Analysis cache** — avoids re-fetching Instagram data for the same username
   within a configurable TTL window. Stored in-memory (resets on restart).

2. **History log** — keeps the last N analyses in a deque for the /history
   and /stats endpoints.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Optional

from app.schemas.analysis import AnalyzeResponse, HistoryEntry

_UTC = timezone.utc


class HistoryService:
    def __init__(self, max_size: int = 500, cache_ttl_seconds: int = 3600):
        self._max_size = max_size
        self._ttl = timedelta(seconds=cache_ttl_seconds)
        # history: newest first
        self._history: deque[HistoryEntry] = deque(maxlen=max_size)
        # cache: username → (AnalyzeResponse, stored_at)
        self._cache: dict[str, tuple[AnalyzeResponse, datetime]] = {}

    # ── Cache ─────────────────────────────────────────────────────────────────

    def get_cached(self, username: str) -> Optional[AnalyzeResponse]:
        entry = self._cache.get(username)
        if entry is None:
            return None
        result, stored_at = entry
        if datetime.now(_UTC) - stored_at > self._ttl:
            del self._cache[username]
            return None
        return result

    def set_cache(self, username: str, result: AnalyzeResponse) -> None:
        self._cache[username] = (result, datetime.now(_UTC))

    def invalidate(self, username: str) -> None:
        self._cache.pop(username, None)

    # ── History log ───────────────────────────────────────────────────────────

    def add(self, response: AnalyzeResponse) -> None:
        entry = HistoryEntry(
            username=response.username,
            prediction=response.analysis.prediction,
            risk_score=response.analysis.risk_score,
            risk_level=response.analysis.risk_level,
            confidence=response.analysis.confidence,
            analyzed_at=response.analyzed_at,
            cached=response.cached,
        )
        self._history.appendleft(entry)

    def get_history(self, limit: int = 50) -> list[HistoryEntry]:
        return list(self._history)[:limit]

    def clear(self) -> None:
        self._history.clear()
        self._cache.clear()

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        entries = list(self._history)
        total = len(entries)
        if total == 0:
            return {
                "total_analyses": 0,
                "distribution": {"Human": 0.0, "Bot": 0.0, "Suspicious": 0.0},
                "avg_risk_score": 0.0,
                "high_risk_rate": 0.0,
            }

        preds = [e.prediction for e in entries]
        risks = [e.risk_score for e in entries]

        distribution = {
            label: round(preds.count(label) / total, 4)
            for label in ("Human", "Bot", "Suspicious")
        }
        avg_risk = round(sum(risks) / total, 4)
        high_risk_rate = round(sum(1 for r in risks if r >= 0.65) / total, 4)

        return {
            "total_analyses": total,
            "distribution": distribution,
            "avg_risk_score": avg_risk,
            "high_risk_rate": high_risk_rate,
        }
