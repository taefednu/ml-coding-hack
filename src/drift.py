"""Population stability utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from src.utils import get_logger

LOGGER = get_logger(__name__)


@dataclass
class PSIScore:
    feature: str
    psi: float


def _psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    expected_percents, _ = np.histogram(expected, bins=buckets)
    actual_percents, _ = np.histogram(actual, bins=buckets)
    expected_percents = expected_percents / expected_percents.sum()
    actual_percents = actual_percents / actual_percents.sum()
    psi = np.sum((actual_percents - expected_percents) * np.log((actual_percents + 1e-6) / (expected_percents + 1e-6)))
    return float(psi)


def compute_psi(train: pd.DataFrame, other: pd.DataFrame) -> List[PSIScore]:
    scores: List[PSIScore] = []
    common_cols = [col for col in train.columns if col in other.columns]
    for col in common_cols:
        if train[col].dtype.kind not in {"i", "f"}:
            continue
        psi = _psi(train[col].dropna().values, other[col].dropna().values)
        scores.append(PSIScore(feature=col, psi=psi))
    return scores


def psi_report(train: pd.DataFrame, valid: pd.DataFrame, oot: pd.DataFrame) -> Dict[str, List[Dict[str, float]]]:
    return {
        "train_valid": [vars(score) for score in compute_psi(train, valid)],
        "train_oot": [vars(score) for score in compute_psi(train, oot)],
    }


__all__ = ["PSIScore", "compute_psi", "psi_report"]
