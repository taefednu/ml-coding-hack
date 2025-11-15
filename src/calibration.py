"""Probability calibration utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.metrics import compute_metrics
from src.utils import get_logger

LOGGER = get_logger(__name__)


@dataclass
class CalibrationResult:
    method: str
    metrics: Dict[str, float]


class ProbabilityCalibrator:
    def __init__(self):
        self.method_: str | None = None
        self.platt_: LogisticRegression | None = None
        self.isotonic_: IsotonicRegression | None = None

    def fit(self, y_true: np.ndarray, y_pred: np.ndarray) -> CalibrationResult:
        platt = LogisticRegression(max_iter=1000)
        platt.fit(y_pred.reshape(-1, 1), y_true)
        platt_probs = platt.predict_proba(y_pred.reshape(-1, 1))[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(y_pred, y_true)
        iso_probs = iso.predict(y_pred)

        platt_metrics = compute_metrics(y_true, platt_probs)
        iso_metrics = compute_metrics(y_true, iso_probs)
        better = platt_metrics if platt_metrics.brier <= iso_metrics.brier else iso_metrics
        if better is platt_metrics:
            self.method_ = "platt"
            self.platt_ = platt
        else:
            self.method_ = "isotonic"
            self.isotonic_ = iso
        LOGGER.info("Selected %s calibration", self.method_)
        return CalibrationResult(method=self.method_, metrics={
            "platt_brier": platt_metrics.brier,
            "isotonic_brier": iso_metrics.brier,
        })

    def transform(self, y_pred: np.ndarray) -> np.ndarray:
        if self.method_ == "platt" and self.platt_ is not None:
            return self.platt_.predict_proba(y_pred.reshape(-1, 1))[:, 1]
        if self.method_ == "isotonic" and self.isotonic_ is not None:
            return self.isotonic_.predict(y_pred)
        return y_pred

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "ProbabilityCalibrator":
        return joblib.load(path)


__all__ = ["ProbabilityCalibrator", "CalibrationResult"]
