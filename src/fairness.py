"""Fairness metrics across sensitive groups."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.utils import get_logger

LOGGER = get_logger(__name__)


def _safe_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_pred))


def group_fairness_report(
    df: pd.DataFrame,
    sensitive_cols: List[str],
    target_col: str,
    pred_col: str,
    score_col: str | None = None,
    min_group_size: int = 50,
) -> Dict[str, List[Dict[str, float]]]:
    report: Dict[str, List[Dict[str, float]]] = {}
    for col in sensitive_cols:
        if col not in df.columns:
            continue
        groups = []
        for value, subset in df.groupby(col):
            if subset.shape[0] < min_group_size:
                continue
            auc = _safe_auc(subset[target_col].values, subset[pred_col].values)
            bad_rate = float(subset[target_col].mean()) if subset[target_col].nunique() > 0 else 0.0
            avg_score = float(subset[score_col].mean()) if score_col and score_col in subset.columns else None
            groups.append(
                {
                    "group": str(value),
                    "count": int(subset.shape[0]),
                    "bad_rate": bad_rate,
                    "auc": auc,
                    "avg_score": avg_score,
                }
            )
        if groups:
            report[col] = groups
    return report


__all__ = ["group_fairness_report"]
