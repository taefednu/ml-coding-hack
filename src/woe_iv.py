"""WOE/IV binning utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif

from src.utils import get_logger

LOGGER = get_logger(__name__)


@dataclass
class BinSummary:
    feature: str
    bin_edges: List[float]
    woe: List[float]
    iv: float


class MonotonicBinner:
    """Simple quantile-based binning with monotonic WOE enforcement."""

    def __init__(self, bins: int = 5):
        self.bins = bins
        self.bin_edges_: Dict[str, np.ndarray] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series):
        for col in X.columns:
            series = pd.to_numeric(X[col], errors="coerce")
            valid = series.dropna()
            if valid.nunique() <= 1:
                LOGGER.warning("Skipping column %s for WOE binning (insufficient variance)", col)
                continue
            quantiles = np.linspace(0, 1, self.bins + 1)
            edges = np.nanquantile(valid, quantiles)
            edges = np.unique(edges[~np.isnan(edges)])
            if len(edges) < 2:
                LOGGER.warning("Could not derive monotonic bins for %s", col)
                continue
            self.bin_edges_[col] = edges
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        for col, edges in self.bin_edges_.items():
            out[col] = pd.cut(
                pd.to_numeric(X[col], errors="coerce"),
                bins=edges,
                include_lowest=True,
                duplicates="drop",
            )
        return out


class WOETransformer(BaseEstimator, TransformerMixin):
    def __init__(self, bins: int = 5, min_samples: int = 50):
        self.binner = MonotonicBinner(bins)
        self.min_samples = min_samples
        self.woe_mappings_: Dict[str, Dict[pd.Interval, float]] = {}
        self.iv_: Dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series):  # type: ignore[override]
        binned = self.binner.fit(X, y).transform(X)
        for col in binned.columns:
            df = pd.DataFrame({"bin": binned[col], "target": y})
            stats = df.groupby("bin", dropna=True, observed=False)["target"].agg(["sum", "count"])
            stats["good"] = stats["count"] - stats["sum"]
            stats = stats[(stats["count"] >= self.min_samples) & (stats["sum"] > 0)]
            if stats.empty:
                LOGGER.warning("No valid bins for %s after filtering", col)
                continue
            total_bad = stats["sum"].sum()
            total_good = stats["good"].sum()
            if total_bad == 0 or total_good == 0:
                LOGGER.warning("Imbalanced bins for %s (bad=%s good=%s)", col, total_bad, total_good)
                continue
            woe_map: Dict[pd.Interval, float] = {}
            iv = 0.0
            prev_woe = None
            for interval, row in stats.iterrows():
                bad_rate = row["sum"] / total_bad
                good_rate = row["good"] / total_good if total_good else 1e-6
                woe = np.log((bad_rate + 1e-6) / (good_rate + 1e-6))
                if prev_woe is not None and woe < prev_woe:
                    woe = prev_woe  # enforce monotonicity
                prev_woe = woe
                woe_map[interval] = woe
                iv += (bad_rate - good_rate) * woe
            if not woe_map:
                LOGGER.warning("WOE mapping empty for %s", col)
                continue
            self.woe_mappings_[col] = woe_map
            self.iv_[col] = iv
        return self

    def transform(self, X: pd.DataFrame):  # type: ignore[override]
        binned = self.binner.transform(X)
        out = pd.DataFrame(index=X.index)
        for col, mapping in self.woe_mappings_.items():
            if col not in binned.columns:
                out[col] = 0.0
                continue
            series = binned[col].astype(object)
            encoded = series.map(mapping)
            encoded = pd.to_numeric(encoded, errors="coerce").fillna(0.0)
            out[col] = encoded.astype(float)
        return out

    def get_bin_summaries(self) -> List[BinSummary]:
        summaries: List[BinSummary] = []
        for feature, mapping in self.woe_mappings_.items():
            edges = self.binner.bin_edges_.get(feature)
            bin_edges = edges.tolist() if edges is not None else []
            woe_values = list(mapping.values())
            summaries.append(
                BinSummary(
                    feature=feature,
                    bin_edges=[float(edge) for edge in bin_edges],
                    woe=[float(val) for val in woe_values],
                    iv=float(self.iv_.get(feature, 0.0)),
                )
            )
        return summaries


def iv_ranking(transformer: WOETransformer) -> pd.DataFrame:
    """Export IV values into a sorted dataframe for feature selection."""

    data = [
        {"feature": feature, "iv": iv}
        for feature, iv in transformer.iv_.items()
    ]
    ranked = pd.DataFrame(data).sort_values("iv", ascending=False).reset_index(drop=True)
    return ranked


def correlation_filter(df: pd.DataFrame, threshold: float = 0.9) -> List[str]:
    """Return columns to drop due to high pairwise correlation."""

    if df.empty:
        return []
    numeric = df.select_dtypes(include=["number"])
    corr = numeric.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return to_drop


def mutual_information_ranking(
    X: pd.DataFrame,
    y: pd.Series,
    discrete_features: bool | List[bool] | None = None,
) -> pd.Series:
    """Compute mutual information scores for each feature."""

    if X.empty:
        return pd.Series(dtype=float)
    scores = mutual_info_classif(X.fillna(0), y, discrete_features=discrete_features, random_state=42)
    return pd.Series(scores, index=X.columns).sort_values(ascending=False)


def stability_selection(
    estimator_factory: Callable[[], BaseEstimator],
    X: pd.DataFrame,
    y: pd.Series,
    n_bootstrap: int = 20,
    sample_frac: float = 0.8,
    random_state: int = 42,
) -> pd.Series:
    """Lightweight stability selection via bootstrap resampling."""

    rng = np.random.default_rng(random_state)
    scores = pd.Series(0.0, index=X.columns)
    n_samples = len(X)
    for _ in range(n_bootstrap):
        sample_idx = rng.choice(n_samples, size=max(1, int(sample_frac * n_samples)), replace=True)
        model = estimator_factory()
        model.fit(X.iloc[sample_idx], y.iloc[sample_idx])
        importances = getattr(model, "feature_importances_", None)
        if importances is None:
            continue
        threshold = np.percentile(importances, 75)
        mask = importances >= threshold
        scores.loc[X.columns[mask]] += 1
    stability = scores / max(1, n_bootstrap)
    return stability.sort_values(ascending=False)


__all__ = [
    "MonotonicBinner",
    "WOETransformer",
    "BinSummary",
    "iv_ranking",
    "correlation_filter",
    "mutual_information_ranking",
    "stability_selection",
]
