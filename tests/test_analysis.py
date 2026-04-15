"""
Integration tests for the /api/v1/analyze endpoint.
Uses MockProviders for all platforms (no real API calls) + real ML model.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


class TestHealthEndpoints:
    def test_root(self, client: TestClient):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert "service" in data
        assert "platforms" in data
        assert set(data["platforms"]) == {"instagram", "twitter", "facebook"}

    def test_health(self, client: TestClient):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_model_info(self, client: TestClient):
        resp = client.get("/api/v1/health/model")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ready"
        assert data["architecture"]["num_classes"] == 3
        assert len(data["features"]) == 24


class TestAnalyzeEndpoint:
    def test_analyze_returns_200(self, client: TestClient):
        resp = client.post("/api/v1/analyze", json={"username": "human_test"})
        assert resp.status_code == 200

    def test_response_structure(self, client: TestClient):
        resp = client.post("/api/v1/analyze", json={"username": "human_test"})
        data = resp.json()
        # Top-level keys
        assert "username" in data
        assert "profile" in data
        assert "analysis" in data
        assert "features" in data
        assert "data_quality" in data
        assert "analyzed_at" in data

    def test_analysis_fields(self, client: TestClient):
        resp = client.post("/api/v1/analyze", json={"username": "human_test"})
        analysis = resp.json()["analysis"]
        assert analysis["prediction"] in ("Human", "Bot", "Suspicious")
        assert 0 <= analysis["confidence"] <= 1
        assert 0 <= analysis["risk_score"] <= 1
        assert analysis["risk_level"] in ("Low", "Medium", "High", "Critical")
        assert set(analysis["probabilities"].keys()) == {"human", "bot", "suspicious"}

    def test_probabilities_sum_to_one(self, client: TestClient):
        resp = client.post("/api/v1/analyze", json={"username": "human_test"})
        probs = resp.json()["analysis"]["probabilities"]
        total = sum(probs.values())
        assert abs(total - 1.0) < 0.01

    def test_features_has_24_entries(self, client: TestClient):
        resp = client.post("/api/v1/analyze", json={"username": "human_test"})
        features = resp.json()["features"]
        assert len(features) == 24

    def test_profile_info_present(self, client: TestClient):
        resp = client.post("/api/v1/analyze", json={"username": "human_test"})
        profile = resp.json()["profile"]
        assert profile["username"] == "human_test"
        assert "followers_count" in profile
        assert "is_verified" in profile

    def test_data_quality_completeness(self, client: TestClient):
        resp = client.post("/api/v1/analyze", json={"username": "human_test"})
        dq = resp.json()["data_quality"]
        assert 0 < dq["completeness"] <= 1.0
        assert isinstance(dq["estimated_features"], list)

    def test_nonexistent_user_returns_404(self, client: TestClient):
        resp = client.post("/api/v1/analyze", json={"username": "nonexistent_user_404"})
        assert resp.status_code == 404

    def test_invalid_username_returns_422(self, client: TestClient):
        resp = client.post("/api/v1/analyze", json={"username": "invalid user!"})
        assert resp.status_code == 422

    def test_username_with_at_sign_is_cleaned(self, client: TestClient):
        resp = client.post("/api/v1/analyze", json={"username": "@human_test"})
        assert resp.status_code == 200
        assert resp.json()["username"] == "human_test"

    def test_second_request_is_cached(self, client: TestClient):
        # First request
        client.post("/api/v1/analyze", json={"username": "bot_test"})
        # Second request should be served from cache
        resp = client.post("/api/v1/analyze", json={"username": "bot_test"})
        assert resp.status_code == 200
        assert resp.json()["cached"] is True

    def test_get_cached_analysis(self, client: TestClient):
        client.post("/api/v1/analyze", json={"username": "suspicious_test"})
        resp = client.get("/api/v1/analyze/suspicious_test")
        assert resp.status_code == 200
        assert resp.json()["cached"] is True

    def test_force_refresh_bypasses_cache(self, client: TestClient):
        # First request — gets cached
        client.post("/api/v1/analyze", json={"username": "bot_test"})
        # Second request with force_refresh — should NOT be cached
        resp = client.post(
            "/api/v1/analyze?force_refresh=true",
            json={"username": "bot_test"},
        )
        assert resp.status_code == 200
        assert resp.json()["cached"] is False

    def test_get_uncached_username_returns_404(self, client: TestClient):
        resp = client.get("/api/v1/analyze/never_analyzed_xyz999")
        assert resp.status_code == 404


class TestHistoryEndpoints:
    def test_history_returns_list(self, client: TestClient):
        # Ensure at least one analysis exists
        client.post("/api/v1/analyze", json={"username": "human_test"})
        resp = client.get("/api/v1/history")
        assert resp.status_code == 200
        data = resp.json()
        assert "total" in data
        assert "results" in data
        assert isinstance(data["results"], list)

    def test_history_entry_structure(self, client: TestClient):
        client.post("/api/v1/analyze", json={"username": "human_test"})
        entries = client.get("/api/v1/history").json()["results"]
        assert len(entries) > 0
        entry = entries[0]
        assert "username" in entry
        assert "prediction" in entry
        assert "risk_score" in entry
        assert "analyzed_at" in entry

    def test_stats_after_analyses(self, client: TestClient):
        client.post("/api/v1/analyze", json={"username": "human_test"})
        resp = client.get("/api/v1/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_analyses"] > 0
        assert "distribution" in data
        assert "avg_risk_score" in data

    def test_export_csv(self, client: TestClient):
        client.post("/api/v1/analyze", json={"username": "human_test"})
        resp = client.get("/api/v1/history/export")
        assert resp.status_code == 200
        assert "text/csv" in resp.headers["content-type"]
        # Check CSV has header row
        content = resp.text
        assert "username" in content
        assert "prediction" in content

    def test_clear_history(self, client: TestClient):
        client.post("/api/v1/analyze", json={"username": "human_test"})
        resp = client.delete("/api/v1/history")
        assert resp.status_code == 200
        # Stats should now return 404
        resp = client.get("/api/v1/stats")
        assert resp.status_code == 404


class TestMultiPlatformAnalysis:
    """Tests for multi-platform bot detection (Twitter, Facebook)."""

    @pytest.mark.parametrize("platform", ["twitter", "facebook"])
    def test_analyze_each_platform(self, client: TestClient, platform: str):
        resp = client.post(
            "/api/v1/analyze",
            json={"username": "human_test", "platform": platform},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["platform"] == platform
        assert data["username"] == "human_test"
        assert data["analysis"]["prediction"] in ("Human", "Bot", "Suspicious")

    @pytest.mark.parametrize("platform", ["twitter", "facebook"])
    def test_bot_detection_per_platform(self, client: TestClient, platform: str):
        resp = client.post(
            "/api/v1/analyze",
            json={"username": "bot_test", "platform": platform},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["platform"] == platform
        assert data["analysis"]["prediction"] in ("Bot", "Suspicious")

    def test_instagram_default_platform(self, client: TestClient):
        """When no platform is specified, it should default to instagram."""
        resp = client.post("/api/v1/analyze", json={"username": "human_test"})
        assert resp.status_code == 200
        assert resp.json()["platform"] == "instagram"

    def test_platform_specific_cache(self, client: TestClient):
        """Same username on different platforms should be cached separately."""
        # Analyze on twitter
        resp_tw = client.post(
            "/api/v1/analyze",
            json={"username": "human_test", "platform": "twitter"},
        )
        assert resp_tw.status_code == 200
        # Analyze on facebook — should NOT be a cache hit from twitter
        resp_fb = client.post(
            "/api/v1/analyze",
            json={"username": "human_test", "platform": "facebook"},
        )
        assert resp_fb.status_code == 200
        assert resp_fb.json()["platform"] == "facebook"

    def test_get_cached_by_platform(self, client: TestClient):
        """GET /analyze/{platform}/{username} should return cached result."""
        client.post(
            "/api/v1/analyze",
            json={"username": "suspicious_test", "platform": "twitter"},
        )
        resp = client.get("/api/v1/analyze/twitter/suspicious_test")
        assert resp.status_code == 200
        assert resp.json()["cached"] is True
        assert resp.json()["platform"] == "twitter"

    @pytest.mark.parametrize("platform", ["twitter", "facebook"])
    def test_nonexistent_user_per_platform(self, client: TestClient, platform: str):
        resp = client.post(
            "/api/v1/analyze",
            json={"username": "nonexistent_user_404", "platform": platform},
        )
        assert resp.status_code == 404

    @pytest.mark.parametrize("platform", ["instagram", "twitter", "facebook"])
    def test_24_features_all_platforms(self, client: TestClient, platform: str):
        resp = client.post(
            "/api/v1/analyze",
            json={"username": "human_test", "platform": platform},
        )
        assert resp.status_code == 200
        assert len(resp.json()["features"]) == 24

    @pytest.mark.parametrize("platform", ["instagram", "twitter", "facebook"])
    def test_probabilities_sum_all_platforms(self, client: TestClient, platform: str):
        resp = client.post(
            "/api/v1/analyze",
            json={"username": "human_test", "platform": platform},
        )
        probs = resp.json()["analysis"]["probabilities"]
        total = sum(probs.values())
        assert abs(total - 1.0) < 0.01
