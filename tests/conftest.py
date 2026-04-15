"""
Pytest configuration and shared fixtures.

The TestClient triggers the FastAPI lifespan, which loads the ML model and
sets up MockProviders for all platforms (no real API calls).
Prerequisites: model artifacts must exist — run `python model/train.py` first.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Ensure project root is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def client(monkeypatch_session) -> TestClient:
    """
    Session-scoped TestClient.
    - Uses the real ML model (loaded via lifespan).
    - Uses MockProvider for ALL platforms (no real API calls).
    - A single client instance is shared across the entire test session
      for performance (model load is expensive).
    """
    monkeypatch_session.setenv("INSTAGRAM_PROVIDER", "mock")
    monkeypatch_session.setenv("TWITTER_PROVIDER", "mock")
    monkeypatch_session.setenv("FACEBOOK_PROVIDER", "mock")
    from app.main import app
    # Use TestClient as a context manager so the lifespan (model load) runs.
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


@pytest.fixture(scope="session")
def monkeypatch_session():
    """Session-scoped monkeypatch."""
    from _pytest.monkeypatch import MonkeyPatch
    mp = MonkeyPatch()
    yield mp
    mp.undo()
