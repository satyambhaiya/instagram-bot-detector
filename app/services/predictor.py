"""
Predictor Service
=================
Loads the trained BotDetectorNet model from disk and runs inference.
This is a singleton — the model is loaded once at startup via lifespan.

Usage
-----
    from app.services.predictor import Predictor
    predictor = Predictor()          # load once
    result = predictor.predict(features_dict)
"""

from __future__ import annotations

import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# Ensure project root is on the path so `model.network` resolves correctly
_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from model.network import BotDetectorNet  # noqa: E402

from app.core.exceptions import ModelNotReadyError
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PredictionOutput:
    prediction: Literal["Human", "Bot", "Suspicious"]
    confidence: float
    risk_score: float
    risk_level: Literal["Low", "Medium", "High", "Critical"]
    probabilities: dict[str, float]   # {"Human": 0.9, "Bot": 0.05, "Suspicious": 0.05}


class Predictor:
    """
    Wraps the PyTorch model and scikit-learn scaler.
    Thread-safe for read-only inference.
    """

    def __init__(self, artifacts_dir: Path):
        self._artifacts = artifacts_dir
        self._model: BotDetectorNet | None = None
        self._scaler = None
        self._meta: dict = {}
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load()

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _load(self) -> None:
        model_path = self._artifacts / "best_model.pt"
        scaler_path = self._artifacts / "scaler.pkl"
        meta_path = self._artifacts / "model_meta.json"

        for p in (model_path, scaler_path, meta_path):
            if not p.exists():
                raise ModelNotReadyError(
                    f"Missing model artifact: {p.name}. "
                    "Run `python model/train.py` to generate it."
                )

        with open(meta_path) as f:
            self._meta = json.load(f)

        with open(scaler_path, "rb") as f:
            self._scaler = pickle.load(f)

        self._model = BotDetectorNet(
            input_dim=self._meta["input_dim"],
            hidden=self._meta["hidden"],
            dropout=self._meta["dropout"],
        ).to(self._device)
        self._model.load_state_dict(
            torch.load(model_path, map_location=self._device, weights_only=True)
        )
        self._model.eval()
        logger.info(
            "Model loaded — device=%s | test_accuracy=%.4f",
            self._device,
            self._meta.get("test_metrics", {}).get("accuracy", 0),
        )

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(
        self,
        features: dict[str, float],
        data_completeness: float = 1.0,
    ) -> PredictionOutput:
        """
        Run inference on a single account's feature dictionary.

        Parameters
        ----------
        features : dict[str, float]
            The 24 features produced by `feature_extractor.extract_features`.
        data_completeness : float
            Fraction of features that were directly observed (not estimated).
            Used to penalise confidence when the analysis relies heavily on estimates.
        """
        # Always enforce eval mode before inference — prevents any accidental
        # gradient/dropout state from making results non-deterministic.
        self._model.eval()

        vec = self._preprocess(features)
        probs = self._run(vec)[0]

        label_names: list[str] = self._meta["label_names"]   # ["Human","Bot","Suspicious"]
        cls_idx = int(np.argmax(probs))
        prediction = label_names[cls_idx]
        raw_confidence = float(probs[cls_idx])

        # ── Confidence penalty for low data quality ───────────────────────────
        # When many features are estimated (completeness < 1), the model is
        # working with imputed data. Reduce the reported confidence accordingly
        # so the frontend can communicate uncertainty to the user.
        #   completeness 1.00 → no penalty
        #   completeness 0.85 → ×0.93
        #   completeness 0.70 → ×0.85
        #   completeness < 0.60 → cap confidence at 0.70
        if data_completeness >= 0.95:
            confidence_factor = 1.0
        elif data_completeness >= 0.80:
            confidence_factor = 0.93
        elif data_completeness >= 0.65:
            confidence_factor = 0.85
        else:
            confidence_factor = 0.75

        adjusted_confidence = round(min(raw_confidence * confidence_factor, 0.99), 4)

        # risk = P(Bot) + 0.5 * P(Suspicious), capped at 1.0
        risk = float(probs[1] + 0.5 * probs[2])
        risk = round(min(risk, 1.0), 4)

        return PredictionOutput(
            prediction=prediction,
            confidence=adjusted_confidence,
            risk_score=risk,
            risk_level=_risk_level(risk),
            probabilities={
                label_names[i]: round(float(p), 4) for i, p in enumerate(probs)
            },
        )

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def metadata(self) -> dict:
        return self._meta

    @property
    def device(self) -> str:
        return str(self._device)

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    # ── Private ───────────────────────────────────────────────────────────────

    def _preprocess(self, features: dict[str, float]) -> np.ndarray:
        feat_cols: list[str] = self._meta["feature_cols"]
        log_cols: list[str] = self._meta["log_cols"]

        row = {c: features.get(c, 0.0) for c in feat_cols}
        df = pd.DataFrame([row])[feat_cols]

        for c in log_cols:
            df[c] = np.log1p(df[c])

        return self._scaler.transform(df).astype(np.float32)

    @torch.no_grad()
    def _run(self, vec: np.ndarray) -> np.ndarray:
        # Set seed for full determinism on CPU (no effect on quality, just ensures
        # that repeated calls with the same input always produce identical output)
        torch.manual_seed(0)
        x = torch.tensor(vec, dtype=torch.float32).to(self._device)
        logits = self._model(x)
        return F.softmax(logits, dim=-1).cpu().numpy()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _risk_level(score: float) -> Literal["Low", "Medium", "High", "Critical"]:
    if score < 0.25:
        return "Low"
    if score < 0.50:
        return "Medium"
    if score < 0.75:
        return "High"
    return "Critical"
