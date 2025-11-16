"""Data preprocessing utilities: cleaning, typing and encoders."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.woe_iv import WOETransformer
from src.utils import get_logger

LOGGER = get_logger(__name__)


@dataclass
class ColumnRoles:
    numeric: List[str]
    categorical: List[str]
    datetime: List[str]
    currency: List[str]


def _to_dataframe(X, columns: List[str]) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X.copy()
    return pd.DataFrame(X, columns=columns)


class DateNormalizer(BaseEstimator, TransformerMixin):
    """Parses common RU/UZ formats (dd.mm.yyyy, yyyy-mm-dd)."""

    def __init__(self, drop_original: bool = False):
        self.drop_original = drop_original
        self.columns_: List[str] = []

    def fit(self, X, y=None):  # type: ignore[override]
        self.columns_ = list(getattr(X, "columns", [])) or [f"f_{i}" for i in range(X.shape[1])]
        return self

    def transform(self, X):  # type: ignore[override]
        df = _to_dataframe(X, self.columns_)
        for col in self.columns_:
            parsed = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
            df[col] = parsed.view("int64") // 10**9
        return df.values


class CurrencyNormalizer(BaseEstimator, TransformerMixin):
    """Converts comma decimal separators and strips currency symbols."""

    pattern = re.compile(r"[^0-9,.-]")

    def __init__(self):
        self.columns_: List[str] = []

    def fit(self, X, y=None):  # type: ignore[override]
        self.columns_ = list(getattr(X, "columns", [])) or [f"f_{i}" for i in range(X.shape[1])]
        return self

    def transform(self, X):  # type: ignore[override]
        df = _to_dataframe(X, self.columns_)
        for col in self.columns_:
            series = (
                df[col]
                .astype(str)
                .str.replace(" ", "", regex=False)
                .str.replace(self.pattern, "", regex=True)
                .str.replace(",", ".", regex=False)
            )
            df[col] = pd.to_numeric(series, errors="coerce")
        return df.values


class Capper(BaseEstimator, TransformerMixin):
    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.bounds_: Dict[str, Tuple[float, float]] = {}

    def fit(self, X, y=None):  # type: ignore[override]
        columns = list(getattr(X, "columns", [f"f_{i}" for i in range(X.shape[1])]))
        df = _to_dataframe(X, columns)
        for col in df.columns:
            lower = df[col].quantile(self.lower_quantile)
            upper = df[col].quantile(self.upper_quantile)
            self.bounds_[col] = (lower, upper)
        return self

    def transform(self, X):  # type: ignore[override]
        df = _to_dataframe(X, list(self.bounds_.keys()))
        for col, (lower, upper) in self.bounds_.items():
            df[col] = df[col].clip(lower, upper)
        return df.values


class CVTargetEncoder(BaseEstimator, TransformerMixin):
    """Cross validated target mean encoder safe for temporal splits."""

    def __init__(self, smoothing: float = 10.0, min_samples_leaf: int = 50):
        self.smoothing = smoothing
        self.min_samples_leaf = min_samples_leaf
        self.global_mean_: float = 0.0
        self.mapping_: Dict[str, Dict[str, float]] = {}
        self.columns_: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series):  # type: ignore[override]
        self.columns_ = list(X.columns)
        self.global_mean_ = float(np.mean(y))
        df = X.copy()
        df["__target__"] = y.values
        for col in self.columns_:
            stats = df.groupby(col)["__target__"].agg(["mean", "count"])
            smoothing = 1 / (1 + np.exp(-(stats["count"] - self.min_samples_leaf) / self.smoothing))
            encoded = self.global_mean_ * (1 - smoothing) + stats["mean"] * smoothing
            self.mapping_[col] = encoded.to_dict()
        return self

    def transform(self, X: pd.DataFrame):  # type: ignore[override]
        df = X.copy()
        for col in self.columns_:
            mapping = self.mapping_.get(col, {})
            df[col] = df[col].map(mapping).fillna(self.global_mean_)
        return df.values


def infer_column_roles(df: pd.DataFrame, cfg: Dict[str, any]) -> ColumnRoles:
    numeric = df.select_dtypes(include=["number"]).columns.tolist()
    categorical = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    datetime_candidates = [col for col in df.columns if "date" in col.lower()]
    currency = [col for col in df.columns if any(token in col.lower() for token in ["amt", "amount", "sum", "uzs", "payment"])]

    manual_numeric = cfg.get("numeric_overrides", [])
    manual_categorical = cfg.get("categorical_overrides", [])
    numeric = sorted(set(numeric) | set(manual_numeric))
    categorical = sorted(set(categorical) | set(manual_categorical))

    return ColumnRoles(
        numeric=numeric,
        categorical=categorical,
        datetime=datetime_candidates,
        currency=currency,
    )


def build_preprocessor(df: pd.DataFrame, cfg: Dict[str, any], target: str | None = None) -> ColumnTransformer:
    roles = infer_column_roles(df, cfg)
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("capper", Capper()),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe: List[Tuple[str, BaseEstimator]]
    encoder_type = cfg.get("categorical_encoder", "onehot")
    if encoder_type == "target" and target is not None:
        encoder = CVTargetEncoder(
            smoothing=cfg.get("target_encoding", {}).get("smoothing", 10.0)
        )
        categorical_steps = [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("target_encoder", encoder),
        ]
    else:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        categorical_steps = [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", encoder),
        ]
    categorical_pipe = Pipeline(steps=categorical_steps)

    transformers: List[Tuple[str, TransformerMixin, List[str]]] = []
    if roles.numeric:
        transformers.append(("numeric", numeric_pipe, roles.numeric))
    if roles.categorical:
        transformers.append(("categorical", categorical_pipe, roles.categorical))
    if roles.datetime:
        transformers.append(("datetime", DateNormalizer(), roles.datetime))
    if roles.currency:
        transformers.append(("currency", Pipeline([("currency", CurrencyNormalizer()), ("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), roles.currency))

    if not transformers:
        raise ValueError("No transformers configured")

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor


@dataclass
class PreprocessingConfig:
    target_col: str = "default"
    id_col: str = "customer_ref"
    drop_columns: List[str] = field(default_factory=list)
    numeric_features: Optional[List[str]] = None
    categorical_features: Optional[List[str]] = None
    winsor_limits: Tuple[float, float] = (0.01, 0.99)
    log_skew_threshold: float = 1.0
    rare_category_threshold: float = 0.01
    categorical_encoding: str = "frequency"
    smoothing: float = 20.0
    min_samples_leaf: int = 100
    enable_woe: bool = False
    woe_features: Optional[List[str]] = None
    scale_numeric: bool = False
    keep_id: bool = True


class AdvancedPreprocessor:
    """Industrial-strength preprocessing for credit scoring data."""

    missing_token = "__missing__"
    empty_token = "__empty__"
    other_token = "__other__"

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        self.numeric_cols_: List[str] = []
        self.categorical_cols_: List[str] = []
        self.numeric_stats_: Dict[str, Dict[str, float]] = {}
        self.categorical_allowed_: Dict[str, Set[str]] = {}
        self.category_levels_: Dict[str, List[str]] = {}
        self.frequency_maps_: Dict[str, Dict[str, float]] = {}
        self.target_maps_: Dict[str, Dict[str, float]] = {}
        self.global_target_mean_: float = 0.0
        self.feature_names_: List[str] = []
        self.woe_transformer_: Optional[WOETransformer] = None
        self.class_weights_: Dict[int, float] = {}
        self.fitted_ = False

    def fit(self, df: pd.DataFrame, y: Optional[pd.Series] = None):
        df = df.copy()
        target_col = self.config.target_col
        y_series: Optional[pd.Series] = y
        if y_series is None and target_col in df.columns:
            y_series = df[target_col]
        if y_series is None:
            raise ValueError("Target vector is required to fit AdvancedPreprocessor")
        y_series = pd.Series(y_series).astype(float)
        self.global_target_mean_ = float(y_series.mean())
        self.class_weights_ = self._compute_class_weights(y_series)

        drop_cols = set(self.config.drop_columns or [])
        drop_cols.add(target_col)
        if not self.config.keep_id:
            drop_cols.add(self.config.id_col)
        df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors="ignore")

        self.numeric_cols_ = self._infer_numeric_columns(df)
        self.categorical_cols_ = self._infer_categorical_columns(df)
        self._fit_numeric(df)
        self._fit_categorical(df, y_series)

        if self.config.enable_woe and self.numeric_cols_:
            processed_numeric = self._transform_numeric(df)
            woe_features = self.config.woe_features or self.numeric_cols_
            available = [col for col in woe_features if col in processed_numeric.columns]
            if available:
                self.woe_transformer_ = WOETransformer(bins=5, min_samples=self.config.min_samples_leaf)
                self.woe_transformer_.fit(processed_numeric[available], y_series)
        self.fitted_ = True

        sample = df.head(1000).copy()
        transformed = self.transform(sample)
        self.feature_names_ = list(transformed.columns)
        return self

    def _compute_class_weights(self, y: pd.Series) -> Dict[int, float]:
        counts = y.value_counts()
        total = len(y)
        n_classes = len(counts)
        weights = {int(cls): total / (n_classes * count) for cls, count in counts.items()}
        return weights

    def _infer_numeric_columns(self, df: pd.DataFrame) -> List[str]:
        if self.config.numeric_features:
            cols = [col for col in self.config.numeric_features if col in df.columns]
        else:
            cols = df.select_dtypes(include=["number"]).columns.tolist()
        if self.config.id_col in cols:
            cols.remove(self.config.id_col)
        return cols

    def _infer_categorical_columns(self, df: pd.DataFrame) -> List[str]:
        if self.config.categorical_features:
            cols = [col for col in self.config.categorical_features if col in df.columns]
        else:
            cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        for drop in (self.config.id_col,):
            if drop in cols:
                cols.remove(drop)
        return cols

    def _fit_numeric(self, df: pd.DataFrame) -> None:
        for col in self.numeric_cols_:
            series = pd.to_numeric(df[col], errors="coerce")
            stats = {
                "median": float(series.median()),
                "lower": float(series.quantile(self.config.winsor_limits[0])),
                "upper": float(series.quantile(self.config.winsor_limits[1])),
                "skew": float(series.skew()),
                "shift": 0.0,
                "scale_mean": float(series.mean()),
                "scale_std": float(series.std(ddof=0)),
            }
            stats["log"] = abs(stats["skew"]) >= self.config.log_skew_threshold
            if stats["log"]:
                min_val = float(series.min())
                stats["shift"] = abs(min_val) + 1 if min_val <= 0 else 0.0
            self.numeric_stats_[col] = stats

    def _fit_categorical(self, df: pd.DataFrame, y: pd.Series) -> None:
        for col in self.categorical_cols_:
            series = df[col]
            normalized = series.astype(str).str.strip().str.lower()
            normalized = normalized.where(series.notna(), self.missing_token)
            normalized = normalized.replace("", self.empty_token)
            freq = normalized.value_counts(normalize=True, dropna=False)
            allowed = set(freq[freq >= self.config.rare_category_threshold].index.tolist())
            allowed.update({self.missing_token, self.empty_token})
            cleaned = normalized.where(normalized.isin(allowed), self.other_token)
            freq_clean = cleaned.value_counts(normalize=True, dropna=False)
            self.categorical_allowed_[col] = allowed
            self.category_levels_[col] = sorted(allowed | {self.other_token})
            self.frequency_maps_[col] = freq_clean.to_dict()
            helper = pd.DataFrame({"value": cleaned, "target": y})
            stats = helper.groupby("value")["target"].agg(["sum", "count"])
            encoded = (stats["sum"] + self.config.smoothing * self.global_target_mean_) / (
                stats["count"] + self.config.smoothing
            )
            mapping = encoded.to_dict()
            mapping.setdefault(self.other_token, self.global_target_mean_)
            self.target_maps_[col] = mapping

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("AdvancedPreprocessor must be fitted before calling transform()")
        df = df.copy()
        numeric_df = self._transform_numeric(df)
        categorical_df = self._transform_categorical(df)
        features = pd.concat([numeric_df, categorical_df], axis=1)
        if self.woe_transformer_ is not None:
            woe_cols = [col for col in self.woe_transformer_.woe_mappings_.keys() if col in features.columns]
            if woe_cols:
                woe_values = self.woe_transformer_.transform(features[woe_cols])
                woe_values = woe_values.add_suffix("_woe")
                features = pd.concat([features, woe_values], axis=1)
        if self.config.keep_id and self.config.id_col in df.columns:
            features.insert(0, self.config.id_col, df[self.config.id_col])
        return features

    def fit_transform(self, df: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        return self.fit(df, y=y).transform(df)

    def _transform_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        data: Dict[str, pd.Series] = {}
        for col in self.numeric_cols_:
            stats = self.numeric_stats_[col]
            series = pd.to_numeric(df.get(col), errors="coerce")
            missing_flag = series.isna().astype(int)
            data[f"{col}_was_missing"] = missing_flag
            series = series.fillna(stats["median"])
            series = series.clip(stats["lower"], stats["upper"])
            if stats["log"]:
                series = np.log1p(series + stats["shift"])
            if self.config.scale_numeric and stats["scale_std"]:
                series = (series - stats["scale_mean"]) / (stats["scale_std"] + 1e-6)
            data[col] = series
        if not data:
            return pd.DataFrame(index=df.index)
        return pd.DataFrame(data, index=df.index)

    def _normalize_category(self, series: pd.Series, allowed: Set[str]) -> pd.Series:
        normalized = series.astype(str).str.strip().str.lower()
        normalized = normalized.where(series.notna(), self.missing_token)
        normalized = normalized.replace("", self.empty_token)
        normalized = normalized.where(normalized.isin(allowed), self.other_token)
        return normalized

    def _transform_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        data: Dict[str, pd.Series] = {}
        encoding = self.config.categorical_encoding.lower()
        for col in self.categorical_cols_:
            allowed = self.categorical_allowed_[col]
            series = df[col] if col in df.columns else pd.Series(self.missing_token, index=df.index)
            normalized = self._normalize_category(series, allowed)
            data[f"{col}_was_missing"] = (normalized == self.missing_token).astype(int)
            if encoding == "onehot":
                categories = self.category_levels_[col]
                cat = pd.Categorical(normalized, categories=categories)
                dummies = pd.get_dummies(cat, prefix=col)
                expected = [f"{col}_{category}" for category in cat.categories]
                dummies = dummies.reindex(columns=expected, fill_value=0)
                for dummy_col in dummies.columns:
                    data[dummy_col] = dummies[dummy_col]
            elif encoding == "frequency":
                freq_map = self.frequency_maps_[col]
                encoded = normalized.map(freq_map).fillna(freq_map.get(self.other_token, 0.0))
                data[f"{col}_freq"] = encoded
            elif encoding == "target":
                mapping = self.target_maps_[col]
                encoded = normalized.map(mapping).fillna(self.global_target_mean_)
                data[f"{col}_target"] = encoded
            else:
                raise ValueError(f"Unknown categorical encoding '{self.config.categorical_encoding}'")
        if not data:
            return pd.DataFrame(index=df.index)
        return pd.DataFrame(data, index=df.index)


__all__ = [
    "CVTargetEncoder",
    "Capper",
    "CurrencyNormalizer",
    "DateNormalizer",
    "ColumnRoles",
    "build_preprocessor",
    "infer_column_roles",
    "AdvancedPreprocessor",
    "PreprocessingConfig",
]
