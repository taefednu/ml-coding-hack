"""Explainability utilities: champion reason codes and challenger SHAP."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils import get_logger, save_json

LOGGER = get_logger(__name__)

try:  # CatBoost is optional
    from catboost import Pool
except Exception:  # pragma: no cover - optional dependency
    Pool = None


def champion_coefficients(model) -> List[Dict[str, float]]:
    coef = model.estimator.coef_.ravel()
    return [
        {"feature": feature, "coefficient": float(weight)}
        for feature, weight in zip(model.feature_cols, coef)
    ]


def champion_reason_codes(model, df: pd.DataFrame, top_k: int = 3, max_samples: int = 200) -> List[Dict[str, object]]:
    sample = df.head(max_samples)
    woe = model.extras.get("woe")
    if woe is None:
        return []
    transformed = woe.transform(sample[model.feature_cols]).fillna(0.0)
    transformed_values = transformed.to_numpy(dtype=float, copy=False)
    coefficients = model.estimator.coef_.ravel()
    contributions = transformed_values * coefficients
    reason_codes: List[Dict[str, object]] = []
    for idx, row in enumerate(sample.index):
        contrib_row = contributions[idx]
        top_idx = np.argsort(-np.abs(contrib_row))[:top_k]
        factors = [
            {
                "feature": model.feature_cols[j],
                "contribution": float(contrib_row[j]),
            }
            for j in top_idx
        ]
        reason_codes.append({"row": int(idx), "index": int(row), "factors": factors})
    return reason_codes


def export_champion_explainability(model, df: pd.DataFrame, cfg: Dict[str, int], path: Path) -> None:
    payload = {
        "intercept": float(model.estimator.intercept_[0]),
        "coefficients": champion_coefficients(model),
        "reason_codes": champion_reason_codes(
            model,
            df,
            top_k=cfg.get("reason_codes_top_k", 3),
            max_samples=cfg.get("reason_codes_max_samples", 200),
        ),
    }
    save_json(payload, path)
    LOGGER.info("Champion explainability saved to %s", path)


def challenger_shap_summary(model, df: pd.DataFrame, cfg: Dict[str, int], path: Path) -> None:
    sample = df[model.feature_cols].head(cfg.get("shap_max_samples", 500))
    importance: List[Dict[str, float]] = []
    if model.name == "catboost" and Pool is not None:
        pool = Pool(sample, cat_features=model.extras.get("categorical", []))
        shap_values = model.estimator.get_feature_importance(pool, type="ShapValues")
        shap_contrib = shap_values[:, :-1]
        mean_abs = np.mean(np.abs(shap_contrib), axis=0)
        importance = [
            {"feature": feat, "mean_abs_shap": float(val)}
            for feat, val in sorted(
                zip(model.feature_cols, mean_abs), key=lambda x: x[1], reverse=True
            )
        ]
    elif hasattr(model.estimator, "feature_importances_"):
        importances = model.estimator.feature_importances_
        importance = [
            {"feature": feat, "importance": float(val)}
            for feat, val in sorted(
                zip(model.feature_cols, importances), key=lambda x: x[1], reverse=True
            )
        ]
    else:
        LOGGER.warning("Challenger %s lacks SHAP/importance hooks", model.name)
    save_json({"feature_importance": importance}, path)
    LOGGER.info("Challenger explainability saved to %s", path)


__all__ = [
    "champion_coefficients",
    "champion_reason_codes",
    "export_champion_explainability",
    "challenger_shap_summary",
]
