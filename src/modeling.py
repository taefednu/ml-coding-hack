"""Model training routines with Champion/Challenger abstraction."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterSampler, TimeSeriesSplit
from sklearn.pipeline import Pipeline

from src.metrics import compute_metrics, report_to_dict
from src.preprocessing import build_preprocessor
from src.woe_iv import WOETransformer
from src.utils import get_logger

LOGGER = get_logger(__name__)

try:  # optional dependencies
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier, Pool
except Exception:  # pragma: no cover
    CatBoostClassifier = None
    Pool = None


@dataclass
class TrainedModel:
    name: str
    estimator: Any
    metrics: Dict[str, float]
    feature_cols: List[str]
    extras: Dict[str, Any]

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if self.name == "logistic_woe" and "woe" in self.extras:
            woe = self.extras["woe"]
            transformed = woe.transform(df[self.feature_cols])
            return self.estimator.predict_proba(transformed)[:, 1]
        return self.estimator.predict_proba(df[self.feature_cols])[:, 1]


def _parse_dates(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", dayfirst=True)
    if parsed.notna().any():
        return parsed
    series_str = series.astype(str)
    formats = ["%Y-%m-%d", "%Y/%m/%d", "%Y%m%d", "%Y"]
    for fmt in formats:
        parsed = pd.to_datetime(series_str, format=fmt, errors="coerce")
        if parsed.notna().any():
            return parsed
    return parsed


def split_by_time(df: pd.DataFrame, date_col: str, train_end: str, valid_end: str) -> Dict[str, pd.DataFrame]:
    df = df.copy()
    raw_dates = df[date_col]
    parsed_dates = _parse_dates(raw_dates)
    if parsed_dates.isna().all():
        raise ValueError(f"Unable to parse date column {date_col} for temporal splits")
    df[date_col] = parsed_dates
    train_end_ts = pd.to_datetime(train_end)
    valid_end_ts = pd.to_datetime(valid_end)
    train = df[df[date_col] <= train_end_ts]
    valid = df[(df[date_col] > train_end_ts) & (df[date_col] <= valid_end_ts)]
    oot = df[df[date_col] > valid_end_ts]
    if valid.empty or oot.empty:
        LOGGER.warning(
            "Configured time thresholds produced empty validation/OOT slices. Falling back to quantile split on %s",
            date_col,
        )
        sorted_df = df.sort_values(date_col)
        n = len(sorted_df)
        train_end_idx = max(1, int(n * 0.6))
        valid_end_idx = max(train_end_idx + 1, int(n * 0.8))
        valid_end_idx = min(valid_end_idx, n - 1) if n > 2 else n
        train = sorted_df.iloc[:train_end_idx]
        valid = sorted_df.iloc[train_end_idx:valid_end_idx]
        oot = sorted_df.iloc[valid_end_idx:]
        if valid.empty and not oot.empty:
            valid = oot.iloc[:1]
            oot = oot.iloc[1:]
        if oot.empty and not valid.empty:
            oot = valid.iloc[-1:]
            valid = valid.iloc[:-1]
    return {"train": train, "valid": valid, "oot": oot}


def _cv_splits(
    df: pd.DataFrame,
    target: str,
    date_col: str,
    group_col: Optional[str],
    n_splits: int,
) -> Iterator[Tuple[pd.DataFrame, pd.DataFrame]]:
    df = df.sort_values(date_col)
    if n_splits <= 1:
        return iter([])
    if group_col and group_col in df.columns:
        group_dates = (
            df[[group_col, date_col]]
            .dropna()
            .groupby(group_col)[date_col]
            .min()
            .sort_values()
        )
        splitter = TimeSeriesSplit(n_splits=n_splits)
        for train_idx, val_idx in splitter.split(group_dates.values):
            train_groups = group_dates.index[train_idx]
            val_groups = group_dates.index[val_idx]
            train_mask = df[group_col].isin(train_groups)
            val_mask = df[group_col].isin(val_groups)
            yield df.loc[train_mask], df.loc[val_mask]
    else:
        splitter = TimeSeriesSplit(n_splits=n_splits)
        for train_idx, val_idx in splitter.split(df):
            yield df.iloc[train_idx], df.iloc[val_idx]


def _random_sampler(space: Dict[str, Tuple[float, float]], n_trials: int) -> ParameterSampler:
    dist = {}
    for param, bounds in space.items():
        low, high = bounds
        if isinstance(low, int) and isinstance(high, int):
            dist[param] = range(low, high + 1)
        else:
            dist[param] = np.linspace(float(low), float(high), num=20)
    return ParameterSampler(dist, n_iter=n_trials, random_state=42)


def train_logistic_woe(train: pd.DataFrame, valid: pd.DataFrame, target: str, cfg: Dict[str, Any]) -> TrainedModel:
    feature_cols = [col for col in train.columns if col != target]
    numeric_cols = [col for col in feature_cols if train[col].dtype.kind in {"i", "f"}]
    woe_cols = cfg.get("woe_features") or numeric_cols
    woe_cols = [col for col in woe_cols if col in numeric_cols]
    woe = WOETransformer(bins=cfg.get("bins", 5))
    X_train = train[woe_cols]
    y_train = train[target]
    X_valid = valid[woe_cols]
    y_valid = valid[target]
    woe.fit(X_train, y_train)
    if not woe.woe_mappings_:
        raise ValueError("WOE transformer did not produce any valid features; check input data")
    lr = LogisticRegression(max_iter=500, class_weight="balanced")
    lr.fit(woe.transform(X_train), y_train)
    if X_valid.empty:
        LOGGER.warning("Validation slice is empty; metrics will be computed on training set")
        valid_data = woe.transform(X_train)
        valid_target = y_train
    else:
        valid_data = woe.transform(X_valid)
        valid_target = y_valid
    valid_pred = lr.predict_proba(valid_data)[:, 1]
    metrics = report_to_dict(compute_metrics(valid_target.values, valid_pred))
    LOGGER.info("Champion Logistic+WOE valid ROC-AUC %.4f", metrics["roc_auc"])
    return TrainedModel(
        name="logistic_woe",
        estimator=lr,
        metrics=metrics,
        feature_cols=woe_cols,
        extras={"woe": woe},
    )


def train_lgbm(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    target: str,
    cfg: Dict[str, Any],
    date_col: str,
    group_col: Optional[str],
    cv_folds: int,
) -> Optional[TrainedModel]:
    if LGBMClassifier is None:
        LOGGER.warning("LightGBM not available, skipping")
        return None
    feature_cols = [col for col in train.columns if col != target]
    preprocessor = build_preprocessor(train[feature_cols], cfg.get("preprocessing", {}), target)
    clf = LGBMClassifier(
        n_estimators=cfg.get("n_estimators", 800),
        learning_rate=cfg.get("learning_rate", 0.05),
        max_depth=cfg.get("max_depth", -1),
        subsample=cfg.get("subsample", 0.8),
        colsample_bytree=cfg.get("colsample", 0.8),
        class_weight="balanced",
        n_jobs=-1,
    )
    pipe = Pipeline([("prep", preprocessor), ("model", clf)])
    X_train = train[feature_cols].copy()
    y_train = train[target]
    X_valid = valid[feature_cols].copy()
    y_valid = valid[target]

    tuning_cfg = cfg.get("tuning", {})
    if tuning_cfg.get("enabled"):
        search_space = {
            "model__learning_rate": (tuning_cfg.get("learning_rate", [0.01, 0.1])[0], tuning_cfg.get("learning_rate", [0.01, 0.1])[1]),
            "model__num_leaves": (tuning_cfg.get("num_leaves", [32, 128])[0], tuning_cfg.get("num_leaves", [32, 128])[1]),
        }
        sampler = _random_sampler(search_space, tuning_cfg.get("n_trials", 5))
        best_auc = -math.inf
        best_params = {}
        for params in sampler:
            pipe.set_params(**params)
            pipe.fit(X_train, y_train)
            preds = pipe.predict_proba(X_valid)[:, 1]
            auc = report_to_dict(compute_metrics(y_valid.values, preds))["roc_auc"]
            if auc > best_auc:
                best_auc = auc
                best_params = params
        if best_params:
            LOGGER.info("LightGBM tuning best params: %s", best_params)
            pipe.set_params(**best_params)

    pipe.fit(X_train, y_train)
    preds = pipe.predict_proba(X_valid)[:, 1]
    metrics = report_to_dict(compute_metrics(y_valid.values, preds))

    if cv_folds > 1:
        cv_scores = []
        for fold, (cv_train, cv_valid) in enumerate(_cv_splits(train, target, date_col, group_col, cv_folds), start=1):
            pipe.fit(cv_train[feature_cols], cv_train[target])
            fold_pred = pipe.predict_proba(cv_valid[feature_cols])[:, 1]
            fold_auc = report_to_dict(compute_metrics(cv_valid[target].values, fold_pred))["roc_auc"]
            cv_scores.append({"fold": fold, "roc_auc": fold_auc})
        if cv_scores:
            metrics["cv_roc_auc_mean"] = float(np.mean([fold["roc_auc"] for fold in cv_scores]))
    LOGGER.info("LightGBM valid ROC-AUC %.4f", metrics["roc_auc"])
    return TrainedModel(
        name="lightgbm",
        estimator=pipe,
        metrics=metrics,
        feature_cols=feature_cols,
        extras={},
    )


def train_catboost(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    target: str,
    cfg: Dict[str, Any],
    date_col: str,
    group_col: Optional[str],
    cv_folds: int,
) -> Optional[TrainedModel]:
    if CatBoostClassifier is None:
        LOGGER.warning("CatBoost not available, skipping")
        return None
    feature_cols = [col for col in train.columns if col != target]
    categorical = [col for col in feature_cols if train[col].dtype == "object"]
    clf = CatBoostClassifier(
        iterations=cfg.get("iterations", 2000),
        learning_rate=cfg.get("learning_rate", 0.05),
        depth=cfg.get("depth", 6),
        l2_leaf_reg=cfg.get("l2_leaf_reg", 3.0),
        loss_function="Logloss",
        eval_metric="AUC",
        auto_class_weights="Balanced",
        verbose=False,
    )
    X_train = train[feature_cols].copy()
    y_train = train[target]
    X_valid = valid[feature_cols].copy()
    y_valid = valid[target]
    if categorical:
        X_train[categorical] = X_train[categorical].fillna("missing")
        X_valid[categorical] = X_valid[categorical].fillna("missing")

    tuning_cfg = cfg.get("tuning", {})
    if tuning_cfg.get("enabled"):
        sampler = _random_sampler(
            {
                "learning_rate": (tuning_cfg.get("learning_rate", [0.02, 0.08])[0], tuning_cfg.get("learning_rate", [0.02, 0.08])[1]),
                "depth": (tuning_cfg.get("depth", [4, 8])[0], tuning_cfg.get("depth", [4, 8])[1]),
                "l2_leaf_reg": (tuning_cfg.get("l2_leaf_reg", [2.0, 10.0])[0], tuning_cfg.get("l2_leaf_reg", [2.0, 10.0])[1]),
            },
            tuning_cfg.get("n_trials", 5),
        )
        best_auc = -math.inf
        best_params = {}
        for params in sampler:
            trial_clf = clf.copy()
            trial_clf.set_params(
                learning_rate=float(params.get("learning_rate", trial_clf.get_param("learning_rate"))),
                depth=int(params.get("depth", trial_clf.get_param("depth"))),
                l2_leaf_reg=float(params.get("l2_leaf_reg", trial_clf.get_param("l2_leaf_reg"))),
            )
            trial_clf.fit(
                X_train,
                y_train,
                eval_set=(X_valid, y_valid),
                cat_features=categorical,
                use_best_model=True,
            )
            preds = trial_clf.predict_proba(X_valid)[:, 1]
            auc = report_to_dict(compute_metrics(y_valid.values, preds))["roc_auc"]
            if auc > best_auc:
                best_auc = auc
                best_params = trial_clf.get_params()
        if best_params:
            LOGGER.info("CatBoost tuning selected params: lr=%.4f depth=%d l2=%.2f", best_params["learning_rate"], best_params["depth"], best_params["l2_leaf_reg"])
            clf.set_params(
                learning_rate=best_params["learning_rate"],
                depth=best_params["depth"],
                l2_leaf_reg=best_params["l2_leaf_reg"],
            )

    clf.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=categorical, use_best_model=True)
    preds = clf.predict_proba(X_valid)[:, 1]
    metrics = report_to_dict(compute_metrics(y_valid.values, preds))

    if cv_folds > 1:
        cv_scores = []
        for fold, (cv_train, cv_valid) in enumerate(_cv_splits(train, target, date_col, group_col, cv_folds), start=1):
            pool_train = Pool(cv_train[feature_cols], label=cv_train[target], cat_features=categorical)
            pool_valid = Pool(cv_valid[feature_cols], label=cv_valid[target], cat_features=categorical)
            clf.fit(pool_train, eval_set=pool_valid, use_best_model=False)
            fold_pred = clf.predict_proba(cv_valid[feature_cols])[:, 1]
            fold_auc = report_to_dict(compute_metrics(cv_valid[target].values, fold_pred))["roc_auc"]
            cv_scores.append({"fold": fold, "roc_auc": fold_auc})
        if cv_scores:
            metrics["cv_roc_auc_mean"] = float(np.mean([fold["roc_auc"] for fold in cv_scores]))
    LOGGER.info("CatBoost valid ROC-AUC %.4f", metrics["roc_auc"])
    return TrainedModel(
        name="catboost",
        estimator=clf,
        metrics=metrics,
        feature_cols=feature_cols,
        extras={"categorical": categorical},
    )


__all__ = [
    "TrainedModel",
    "split_by_time",
    "train_catboost",
    "train_lgbm",
    "train_logistic_woe",
]
