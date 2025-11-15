"""Custom metrics for credit risk evaluation."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from src.utils import get_logger

LOGGER = get_logger(__name__)


@dataclass
class MetricReport:
    roc_auc: float
    gini: float
    pr_auc: float
    ks: float
    logloss: float
    brier: float
    ece: float


def ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    order = np.argsort(y_prob)
    y_true = y_true[order]
    y_prob = y_prob[order]
    cum_pos = np.cumsum(y_true) / max(y_true.sum(), 1)
    cum_neg = np.cumsum(1 - y_true) / max((1 - y_true).sum(), 1)
    return float(np.max(np.abs(cum_pos - cum_neg)))


def reliability_curve(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Dict[str, List[float]]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    prob_true: List[float] = []
    prob_pred: List[float] = []
    for i in range(n_bins):
        mask = binids == i
        if not np.any(mask):
            continue
        prob_true.append(float(y_true[mask].mean()))
        prob_pred.append(float(y_prob[mask].mean()))
    return {"true": prob_true, "pred": prob_pred}


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    total = len(y_true)
    ece = 0.0
    for i in range(n_bins):
        mask = binids == i
        if not np.any(mask):
            continue
        weight = np.sum(mask) / total
        ece += weight * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


def lift_curve(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Dict[str, List[float]]:
    order = np.argsort(y_prob)[::-1]
    y_true = y_true[order]
    y_prob = y_prob[order]
    step = max(len(y_true) // n_bins, 1)
    lifts: List[float] = []
    gains: List[float] = []
    baseline = y_true.mean() + 1e-6
    for i in range(step, len(y_true) + 1, step):
        subset = y_true[:i]
        gains.append(float(subset.mean()))
        lifts.append(float(subset.mean() / baseline))
    return {"lift": lifts, "gain": gains}


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> MetricReport:
    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))
    gini = 2 * roc_auc - 1
    ks = ks_statistic(y_true, y_prob)
    y_prob_clipped = np.clip(y_prob, 1e-15, 1 - 1e-15)
    logloss = float(log_loss(y_true, y_prob_clipped))
    brier = float(brier_score_loss(y_true, y_prob))
    ece = expected_calibration_error(y_true, y_prob)
    return MetricReport(roc_auc, gini, pr_auc, ks, logloss, brier, ece)


def report_to_dict(report: MetricReport) -> Dict[str, float]:
    return {
        "roc_auc": report.roc_auc,
        "gini": report.gini,
        "pr_auc": report.pr_auc,
        "ks": report.ks,
        "logloss": report.logloss,
        "brier": report.brier,
        "ece": report.ece,
    }


__all__ = [
    "MetricReport",
    "compute_metrics",
    "expected_calibration_error",
    "ks_statistic",
    "lift_curve",
    "reliability_curve",
    "report_to_dict",
]
