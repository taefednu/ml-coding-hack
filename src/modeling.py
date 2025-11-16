"""Model training routines with Champion/Challenger abstraction."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterSampler, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

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

try:
    import optuna
except Exception:  # pragma: no cover
    optuna = None

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None


LEAK_PATTERNS = [
    "default",
    "default_flag",
    "is_default",
    "label",
    "dpd",
    "delinq",
    "overdue",
    "chargeoff",
    "charged_off",
    "writeoff",
    "ground_truth",
    "bad_flag",
    "status_after",
    "collection",
    "recovery",
]
DATE_PATTERNS = ["date", "timestamp", "payoff_date", "report_date"]
SERVICE_COLUMNS = {"row_id", "index", "__index_level_0__"}


def is_safe_feature(name: str, target: str, forbidden_ids: Optional[List[str]] = None) -> bool:
    lower = name.lower()
    if name == target:
        return False
    if forbidden_ids and any(name == candidate or lower == candidate.lower() for candidate in forbidden_ids):
        return False
    if lower in SERVICE_COLUMNS:
        return False
    if lower == "y" or lower.startswith("y_") or lower.endswith("_y"):
        return False
    if any(pattern in lower for pattern in LEAK_PATTERNS):
        return False
    if any(pattern in lower for pattern in DATE_PATTERNS):
        return False
    return True


def _filter_features(columns: List[str], target: str, forbidden_ids: Optional[List[str]] = None) -> List[str]:
    safe = [col for col in columns if col != target and is_safe_feature(col, target, forbidden_ids)]
    if not safe:
        raise ValueError("No safe features remaining after leakage filtering")
    return safe


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
        if self.name == "ensemble" and "base_models" in self.extras:
            preds = np.zeros(len(df), dtype=float)
            weights = self.extras.get("weights", [])
            base_models = self.extras.get("base_models", [])
            for weight, model in zip(weights, base_models):
                preds += weight * model.predict_proba(df)
            return preds
        if self.name == "stacking" and "base_models" in self.extras:
            base_models = self.extras.get("base_models", [])
            base_preds = np.column_stack([m.predict_proba(df) for m in base_models])
            return self.estimator.predict_proba(base_preds)[:, 1]
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
        LOGGER.info(
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


def _is_metrics_suspicious(metrics: Dict[str, float], cfg: Dict[str, Any]) -> bool:
    if not cfg:
        return False
    if metrics.get("roc_auc", 0.0) >= cfg.get("auc", 0.9999):
        return True
    if metrics.get("ks", 0.0) >= cfg.get("ks", 0.9999):
        return True
    if metrics.get("pr_auc", 0.0) >= cfg.get("pr_auc", 0.9999):
        return True
    if metrics.get("brier", 1.0) <= cfg.get("brier", 1e-4):
        return True
    return False


def _shuffled_auc_pipe(
    pipe: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    sample_size: int,
) -> float:
    if sample_size and len(X_train) > sample_size:
        sample = X_train.sample(sample_size, random_state=42)
        y_sample = y_train.loc[sample.index]
    else:
        sample = X_train
        y_sample = y_train
    shuffled = y_sample.sample(frac=1.0, random_state=42)
    sanity_pipe = clone(pipe)
    sanity_pipe.fit(sample, shuffled)
    preds = sanity_pipe.predict_proba(X_valid)[:, 1]
    return float(roc_auc_score(y_valid, preds))


def _shuffled_auc_catboost(
    clf: CatBoostClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    categorical: List[str],
    sample_size: int,
) -> float:
    if sample_size and len(X_train) > sample_size:
        sample = X_train.sample(sample_size, random_state=42)
        y_sample = y_train.loc[sample.index]
    else:
        sample = X_train
        y_sample = y_train
    shuffled = y_sample.sample(frac=1.0, random_state=42)
    sanity_clf = clf.copy()
    pool_train = Pool(sample, label=shuffled, cat_features=categorical) if categorical else Pool(sample, label=shuffled)
    pool_valid = Pool(X_valid, label=y_valid, cat_features=categorical) if categorical else Pool(X_valid, label=y_valid)
    sanity_clf.fit(pool_train, eval_set=pool_valid, use_best_model=False, verbose=False)
    preds = sanity_clf.predict_proba(X_valid)[:, 1]
    return float(roc_auc_score(y_valid, preds))


def train_logistic_woe(train: pd.DataFrame, valid: pd.DataFrame, target: str, cfg: Dict[str, Any]) -> TrainedModel:
    feature_cols = [col for col in train.columns if col != target]
    numeric_cols = [col for col in feature_cols if train[col].dtype.kind in {"i", "f"}]
    woe_cols = cfg.get("woe_features") or numeric_cols
    woe_cols = [col for col in woe_cols if col in numeric_cols]
    woe_cols = _filter_features(woe_cols, target)
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
        LOGGER.info("Validation slice is empty; metrics will be computed on training set")
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


def _optuna_tune_lgbm(
    pipe: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    tuning_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Use Optuna for efficient hyperparameter tuning of LightGBM."""
    if optuna is None:
        LOGGER.info("Optuna not available, falling back to random search")
        return {}
    
    def objective(trial):
        params = {
            "model__learning_rate": trial.suggest_float(
                "learning_rate",
                tuning_cfg.get("learning_rate", [0.01, 0.08])[0],
                tuning_cfg.get("learning_rate", [0.01, 0.08])[1],
                log=True
            ),
            "model__num_leaves": trial.suggest_int(
                "num_leaves",
                tuning_cfg.get("num_leaves", [32, 192])[0],
                tuning_cfg.get("num_leaves", [32, 192])[1]
            ),
            "model__min_child_samples": trial.suggest_int(
                "min_child_samples",
                tuning_cfg.get("min_child_samples", [50, 500])[0],
                tuning_cfg.get("min_child_samples", [50, 500])[1]
            ),
            "model__subsample": trial.suggest_float(
                "subsample",
                tuning_cfg.get("subsample", [0.7, 1.0])[0],
                tuning_cfg.get("subsample", [0.7, 1.0])[1]
            ),
            "model__colsample_bytree": trial.suggest_float(
                "colsample",
                tuning_cfg.get("colsample", [0.6, 1.0])[0],
                tuning_cfg.get("colsample", [0.6, 1.0])[1]
            ),
            "model__reg_lambda": trial.suggest_float(
                "reg_lambda",
                tuning_cfg.get("reg_lambda", [0.0, 50.0])[0],
                tuning_cfg.get("reg_lambda", [0.0, 50.0])[1]
            ),
            "model__reg_alpha": trial.suggest_float(
                "reg_alpha",
                tuning_cfg.get("reg_alpha", [0.0, 20.0])[0],
                tuning_cfg.get("reg_alpha", [0.0, 20.0])[1]
            ),
        }
        trial_pipe = clone(pipe)
        trial_pipe.set_params(**params)
        trial_pipe.fit(X_train, y_train)
        preds = trial_pipe.predict_proba(X_valid)[:, 1]
        auc = roc_auc_score(y_valid, preds)
        return auc
    
    n_trials = tuning_cfg.get("n_trials", 50)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10) if optuna.pruners else None
    study = optuna.create_study(
        direction="maximize", 
        study_name="lgbm_tuning",
        pruner=pruner,
        sampler=optuna.samplers.TPESampler(seed=42) if optuna.samplers else None
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    best_params = {
        "model__learning_rate": study.best_params["learning_rate"],
        "model__num_leaves": study.best_params["num_leaves"],
        "model__min_child_samples": study.best_params["min_child_samples"],
        "model__subsample": study.best_params["subsample"],
        "model__colsample_bytree": study.best_params["colsample"],
        "model__reg_lambda": study.best_params["reg_lambda"],
        "model__reg_alpha": study.best_params["reg_alpha"],
    }
    LOGGER.info("Optuna tuning best AUC: %.4f, params: %s", study.best_value, best_params)
    return best_params


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
        LOGGER.info("LightGBM not available, skipping")
        return None
    feature_cols = _filter_features([col for col in train.columns if col != target], target)
    preprocessor = build_preprocessor(train[feature_cols], cfg.get("preprocessing", {}), target)
    
    # Optimize class weight if enabled
    class_weight = "balanced"
    if cfg.get("optimize_class_weight", False):
        pos_count = train[target].sum()
        neg_count = len(train) - pos_count
        scale_pos_weight = neg_count / (pos_count + 1e-6)
        class_weight = None
    else:
        scale_pos_weight = None
    
    # Get monotone constraints if specified
    monotone_constraints = cfg.get("monotone_constraints")
    if monotone_constraints and isinstance(monotone_constraints, dict):
        # Convert feature names to indices in preprocessed space
        try:
            prep_sample = preprocessor.transform(train[feature_cols].head(1))
            # Map feature names to monotone constraints (-1, 0, 1)
            mono_list = [0] * prep_sample.shape[1]
            # Simplified: assume all numeric features get monotone constraints
            if "all_numeric" in monotone_constraints:
                direction = 1 if monotone_constraints["all_numeric"] == "increasing" else -1
                # Approximate: assume first features are numeric
                numeric_cols = [col for col in feature_cols if train[col].dtype in ["int64", "float64"]]
                prep_feature_names = preprocessor.get_feature_names_out()
                for i, name in enumerate(prep_feature_names):
                    if any(nc in name for nc in numeric_cols[:len(numeric_cols)//2]):
                        mono_list[i] = direction
            monotone_constraints = mono_list
        except Exception as e:
            LOGGER.warning("Failed to set monotone constraints: %s", e)
            monotone_constraints = None
    
    clf = LGBMClassifier(
        n_estimators=cfg.get("n_estimators", 800),
        learning_rate=cfg.get("learning_rate", 0.05),
        max_depth=cfg.get("max_depth", -1),
        subsample=cfg.get("subsample", 0.8),
        colsample_bytree=cfg.get("colsample", 0.8),
        class_weight=class_weight,
        scale_pos_weight=scale_pos_weight,
        monotone_constraints=monotone_constraints if monotone_constraints else None,
        n_jobs=-1,
        verbosity=-1,
    )
    pipe = Pipeline([("prep", preprocessor), ("model", clf)])
    X_train = train[feature_cols].copy()
    y_train = train[target]
    X_valid = valid[feature_cols].copy()
    y_valid = valid[target]

    tuning_cfg = cfg.get("tuning", {})
    if tuning_cfg.get("enabled"):
        if tuning_cfg.get("use_optuna", False):
            best_params = _optuna_tune_lgbm(pipe, X_train, y_train, X_valid, y_valid, tuning_cfg)
            if best_params:
                pipe.set_params(**best_params)
        else:
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
    suspicious_cfg = cfg.get("suspicious", {})
    is_suspicious = _is_metrics_suspicious(metrics, suspicious_cfg)
    sanity_cfg = cfg.get("sanity_check", {})
    if not is_suspicious and sanity_cfg.get("enabled", True):
        sample_size = int(sanity_cfg.get("sample_size", min(len(X_train), 20000)))
        try:
            shuffled_auc = _shuffled_auc_pipe(pipe, X_train, y_train, X_valid, y_valid, sample_size)
            metrics["sanity_shuffled_auc"] = shuffled_auc
            if shuffled_auc >= sanity_cfg.get("auc_threshold", 0.6):
                LOGGER.warning("LightGBM shuffled-target AUC %.4f indicates potential leakage", shuffled_auc)
                is_suspicious = True
        except Exception as exc:  # pragma: no cover - diagnostic only
            LOGGER.warning("LightGBM sanity check failed: %s", exc)
    metrics["is_suspicious"] = bool(is_suspicious)

    if cv_folds > 1:
        cv_scores = []
        for fold, (cv_train, cv_valid) in enumerate(_cv_splits(train, target, date_col, group_col, cv_folds), start=1):
            cv_pipe = clone(pipe)
            cv_pipe.fit(cv_train[feature_cols], cv_train[target])
            fold_pred = cv_pipe.predict_proba(cv_valid[feature_cols])[:, 1]
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
        extras={"is_suspicious": bool(is_suspicious)},
    )


def train_weighted_ensemble(
    models: List[Optional[TrainedModel]],
    train: pd.DataFrame,
    valid: pd.DataFrame,
    target: str,
    cfg: Optional[Dict[str, Any]] = None,
) -> Optional[TrainedModel]:
    if not cfg or not cfg.get("enabled", True):
        return None
    available: List[TrainedModel] = [m for m in models if m is not None]
    if len(available) < 2:
        return None
    y_valid = valid[target].values
    y_train = train[target].values
    valid_matrix = np.column_stack([m.predict_proba(valid) for m in available])
    train_matrix = np.column_stack([m.predict_proba(train) for m in available])
    if valid_matrix.ndim != 2 or valid_matrix.shape[1] < 2:
        return None
    rng = np.random.default_rng(int(cfg.get("seed", 42)))
    n_trials = int(cfg.get("n_trials", 64))
    min_weight = float(cfg.get("min_weight", 0.0))
    candidates: List[np.ndarray] = []
    n_models = valid_matrix.shape[1]
    candidates.append(np.full(n_models, 1.0 / n_models))
    for idx in range(n_models):
        one_hot = np.zeros(n_models)
        one_hot[idx] = 1.0
        candidates.append(one_hot)
    for _ in range(max(n_trials, 1)):
        weights = rng.dirichlet(np.ones(n_models))
        if min_weight > 0.0:
            weights = np.clip(weights, min_weight, None)
            weights = weights / weights.sum()
        candidates.append(weights)
    best_auc = -math.inf
    best_weights = candidates[0]
    best_valid = valid_matrix @ best_weights
    for weights in candidates:
        combined = valid_matrix @ weights
        auc = roc_auc_score(y_valid, combined)
        if auc > best_auc:
            best_auc = auc
            best_weights = weights
            best_valid = combined
    valid_metrics = report_to_dict(compute_metrics(y_valid, best_valid))
    train_pred = train_matrix @ best_weights
    train_metrics = report_to_dict(compute_metrics(y_train, train_pred))
    valid_metrics["train_roc_auc"] = train_metrics.get("roc_auc")
    suspicious = any(m.extras.get("is_suspicious") for m in available if isinstance(m.extras, dict))
    extras = {
        "base_models": available,
        "weights": best_weights,
        "is_suspicious": suspicious,
    }
    LOGGER.info(
        "Ensemble weights %s produced valid ROC-AUC %.4f",
        np.round(best_weights, 3).tolist(),
        valid_metrics["roc_auc"],
    )
    return TrainedModel(
        name="ensemble",
        estimator=None,
        metrics=valid_metrics,
        feature_cols=[],
        extras=extras,
    )


def _optuna_tune_catboost(
    clf: CatBoostClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    categorical: List[str],
    tuning_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Use Optuna for efficient hyperparameter tuning of CatBoost."""
    if optuna is None:
        LOGGER.info("Optuna not available, falling back to random search")
        return {}
    
    def objective(trial):
        trial_clf = clf.copy()
        trial_clf.set_params(
            learning_rate=trial.suggest_float(
                "learning_rate",
                tuning_cfg.get("learning_rate", [0.02, 0.08])[0],
                tuning_cfg.get("learning_rate", [0.02, 0.08])[1],
                log=True
            ),
            depth=trial.suggest_int("depth", tuning_cfg.get("depth", [4, 8])[0], tuning_cfg.get("depth", [4, 8])[1]),
            l2_leaf_reg=trial.suggest_float(
                "l2_leaf_reg",
                tuning_cfg.get("l2_leaf_reg", [2.0, 10.0])[0],
                tuning_cfg.get("l2_leaf_reg", [2.0, 10.0])[1]
            ),
            subsample=trial.suggest_float(
                "subsample",
                tuning_cfg.get("subsample", [0.7, 1.0])[0],
                tuning_cfg.get("subsample", [0.7, 1.0])[1]
            ) if tuning_cfg.get("subsample") else None,
            colsample_bylevel=trial.suggest_float(
                "colsample_bylevel",
                tuning_cfg.get("colsample_bylevel", [0.6, 1.0])[0],
                tuning_cfg.get("colsample_bylevel", [0.6, 1.0])[1]
            ) if tuning_cfg.get("colsample_bylevel") else None,
        )
        trial_clf.fit(
            X_train, y_train,
            eval_set=(X_valid, y_valid),
            cat_features=categorical if categorical else None,
            use_best_model=True,
            verbose=False
        )
        preds = trial_clf.predict_proba(X_valid)[:, 1]
        auc = roc_auc_score(y_valid, preds)
        return auc
    
    n_trials = tuning_cfg.get("n_trials", 50)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10) if optuna.pruners else None
    study = optuna.create_study(
        direction="maximize",
        study_name="catboost_tuning",
        pruner=pruner,
        sampler=optuna.samplers.TPESampler(seed=42) if optuna.samplers else None
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    best_params = study.best_params.copy()
    LOGGER.info("Optuna tuning best AUC: %.4f, params: %s", study.best_value, best_params)
    return best_params


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
        LOGGER.info("CatBoost not available, skipping")
        return None
    feature_cols = _filter_features([col for col in train.columns if col != target], target)
    categorical = [col for col in feature_cols if train[col].dtype == "object"]
    
    # Optimize class weight if enabled
    auto_class_weights = "Balanced"
    if cfg.get("optimize_class_weight", False):
        pos_count = train[target].sum()
        neg_count = len(train) - pos_count
        class_weights = [neg_count / (pos_count + 1e-6), 1.0]
        auto_class_weights = class_weights
    
    # Get monotone constraints if specified
    monotone_constraints = None
    if cfg.get("monotone_constraints"):
        monotone_dict = cfg.get("monotone_constraints")
        if isinstance(monotone_dict, dict):
            # CatBoost expects a list of 1, -1, or 0 for each numeric feature
            numeric_cols = [col for col in feature_cols if col not in categorical]
            if "all_numeric" in monotone_dict:
                direction = 1 if monotone_dict["all_numeric"] == "increasing" else -1
                monotone_constraints = [direction if col in numeric_cols else 0 for col in feature_cols]
            elif "features" in monotone_dict:
                # Specific feature constraints
                feature_constraints = monotone_dict["features"]
                monotone_constraints = []
                for col in feature_cols:
                    if col in feature_constraints:
                        mono_val = 1 if feature_constraints[col] == "increasing" else -1
                        monotone_constraints.append(mono_val)
                    else:
                        monotone_constraints.append(0)
    
    clf = CatBoostClassifier(
        iterations=cfg.get("iterations", 2000),
        learning_rate=cfg.get("learning_rate", 0.05),
        depth=cfg.get("depth", 6),
        l2_leaf_reg=cfg.get("l2_leaf_reg", 3.0),
        loss_function="Logloss",
        eval_metric="AUC",
        auto_class_weights=auto_class_weights,
        monotone_constraints=monotone_constraints if monotone_constraints else None,
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
        if tuning_cfg.get("use_optuna", False):
            best_params = _optuna_tune_catboost(clf, X_train, y_train, X_valid, y_valid, categorical, tuning_cfg)
            if best_params:
                clf.set_params(
                    learning_rate=best_params["learning_rate"],
                    depth=best_params["depth"],
                    l2_leaf_reg=best_params["l2_leaf_reg"],
                )
                if "subsample" in best_params:
                    clf.set_params(subsample=best_params["subsample"])
                if "colsample_bylevel" in best_params:
                    clf.set_params(colsample_bylevel=best_params["colsample_bylevel"])
        else:
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
    suspicious_cfg = cfg.get("suspicious", {})
    is_suspicious = _is_metrics_suspicious(metrics, suspicious_cfg)
    sanity_cfg = cfg.get("sanity_check", {})
    if not is_suspicious and sanity_cfg.get("enabled", True):
        sample_size = int(sanity_cfg.get("sample_size", min(len(X_train), 20000)))
        try:
            shuffled_auc = _shuffled_auc_catboost(clf, X_train, y_train, X_valid, y_valid, categorical, sample_size)
            metrics["sanity_shuffled_auc"] = shuffled_auc
            if shuffled_auc >= sanity_cfg.get("auc_threshold", 0.6):
                LOGGER.warning("CatBoost shuffled-target AUC %.4f indicates potential leakage", shuffled_auc)
                is_suspicious = True
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("CatBoost sanity check failed: %s", exc)
    metrics["is_suspicious"] = bool(is_suspicious)

    if cv_folds > 1:
        cv_scores = []
        for fold, (cv_train, cv_valid) in enumerate(_cv_splits(train, target, date_col, group_col, cv_folds), start=1):
            fold_clf = clf.copy()
            pool_train = Pool(cv_train[feature_cols], label=cv_train[target], cat_features=categorical)
            pool_valid = Pool(cv_valid[feature_cols], label=cv_valid[target], cat_features=categorical)
            fold_clf.fit(pool_train, eval_set=pool_valid, use_best_model=False)
            fold_pred = fold_clf.predict_proba(cv_valid[feature_cols])[:, 1]
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
        extras={"categorical": categorical, "is_suspicious": bool(is_suspicious)},
    )


def train_xgboost(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    target: str,
    cfg: Dict[str, Any],
    date_col: str,
    group_col: Optional[str],
    cv_folds: int,
) -> Optional[TrainedModel]:
    """Train XGBoost classifier with Optuna tuning."""
    if XGBClassifier is None:
        LOGGER.info("XGBoost not available, skipping")
        return None
    feature_cols = _filter_features([col for col in train.columns if col != target], target)
    preprocessor = build_preprocessor(train[feature_cols], cfg.get("preprocessing", {}), target)
    
    # Optimize class weight if enabled
    scale_pos_weight = cfg.get("scale_pos_weight")
    if cfg.get("optimize_class_weight", False) and scale_pos_weight is None:
        pos_count = train[target].sum()
        neg_count = len(train) - pos_count
        scale_pos_weight = neg_count / (pos_count + 1e-6)
    
    # Get monotone constraints if specified
    monotone_constraints = None
    if cfg.get("monotone_constraints"):
        monotone_dict = cfg.get("monotone_constraints")
        if isinstance(monotone_dict, dict):
            try:
                prep_sample = preprocessor.transform(train[feature_cols].head(1))
                numeric_cols = [col for col in feature_cols if train[col].dtype in ["int64", "float64"]]
                prep_feature_names = preprocessor.get_feature_names_out()
                mono_list = []
                if "all_numeric" in monotone_dict:
                    direction = 1 if monotone_dict["all_numeric"] == "increasing" else -1
                    for name in prep_feature_names:
                        if any(nc in name for nc in numeric_cols[:len(numeric_cols)//2]):
                            mono_list.append(direction)
                        else:
                            mono_list.append(0)
                else:
                    mono_list = [0] * len(prep_feature_names)
                if mono_list:
                    monotone_constraints = tuple(mono_list)
            except Exception as e:
                LOGGER.warning("Failed to set monotone constraints for XGBoost: %s", e)
    
    clf = XGBClassifier(
        n_estimators=cfg.get("n_estimators", 1500),
        learning_rate=cfg.get("learning_rate", 0.05),
        max_depth=cfg.get("max_depth", 6),
        subsample=cfg.get("subsample", 0.8),
        colsample_bytree=cfg.get("colsample", 0.8),
        scale_pos_weight=scale_pos_weight,
        monotone_constraints=monotone_constraints,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    pipe = Pipeline([("prep", preprocessor), ("model", clf)])
    X_train = train[feature_cols].copy()
    y_train = train[target]
    X_valid = valid[feature_cols].copy()
    y_valid = valid[target]

    tuning_cfg = cfg.get("tuning", {})
    if tuning_cfg.get("enabled") and tuning_cfg.get("use_optuna", False) and optuna is not None:
        def objective(trial):
            params = {
                "model__learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
                "model__max_depth": trial.suggest_int("max_depth", 3, 8),
                "model__min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "model__subsample": trial.suggest_float("subsample", 0.7, 1.0),
                "model__colsample_bytree": trial.suggest_float("colsample", 0.6, 1.0),
                "model__reg_lambda": trial.suggest_float("reg_lambda", 0.0, 50.0),
                "model__reg_alpha": trial.suggest_float("reg_alpha", 0.0, 20.0),
            }
            trial_pipe = clone(pipe)
            trial_pipe.set_params(**params)
            trial_pipe.fit(X_train, y_train)
            preds = trial_pipe.predict_proba(X_valid)[:, 1]
            return roc_auc_score(y_valid, preds)
        
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=tuning_cfg.get("n_trials", 50), show_progress_bar=False)
        best_params = {f"model__{k}": v for k, v in study.best_params.items()}
        pipe.set_params(**best_params)
        LOGGER.info("XGBoost Optuna best AUC: %.4f", study.best_value)

    pipe.fit(X_train, y_train)
    preds = pipe.predict_proba(X_valid)[:, 1]
    metrics = report_to_dict(compute_metrics(y_valid.values, preds))
    LOGGER.info("XGBoost valid ROC-AUC %.4f", metrics["roc_auc"])
    return TrainedModel(
        name="xgboost",
        estimator=pipe,
        metrics=metrics,
        feature_cols=feature_cols,
        extras={},
    )


def train_stacking_ensemble(
    models: List[Optional[TrainedModel]],
    train: pd.DataFrame,
    valid: pd.DataFrame,
    target: str,
    cfg: Optional[Dict[str, Any]] = None,
) -> Optional[TrainedModel]:
    """Train stacking ensemble with Logistic Regression meta-learner using out-of-fold predictions."""
    if not cfg or not cfg.get("enabled", True):
        return None
    from sklearn.model_selection import KFold
    from sklearn.linear_model import LogisticRegression
    
    available = [m for m in models if m is not None and m.name != "logistic_woe"]
    if len(available) < 2:
        return None
    
    y_train = train[target].values
    y_valid = valid[target].values
    n_folds = cfg.get("n_folds", 5)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Out-of-fold predictions for training meta-learner
    oof_preds = np.zeros((len(train), len(available)))
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(train)):
        fold_train = train.iloc[train_idx]
        fold_valid = train.iloc[val_idx]
        for model_idx, model in enumerate(available):
            try:
                fold_pred = model.predict_proba(fold_valid)[:, 1]
                oof_preds[val_idx, model_idx] = fold_pred
            except Exception as e:
                LOGGER.warning("Stacking fold %d model %s failed: %s", fold_idx, model.name, e)
    
    # Validation predictions for meta-learner
    valid_preds = np.column_stack([m.predict_proba(valid) for m in available])
    
    # Train meta-learner
    meta_model = LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)
    meta_model.fit(oof_preds, y_train)
    
    # Final predictions
    stacked_pred = meta_model.predict_proba(valid_preds)[:, 1]
    metrics = report_to_dict(compute_metrics(y_valid, stacked_pred))
    LOGGER.info("Stacking ensemble valid ROC-AUC %.4f", metrics["roc_auc"])
    
    return TrainedModel(
        name="stacking",
        estimator=meta_model,
        metrics=metrics,
        feature_cols=[],
        extras={"base_models": available, "n_folds": n_folds},
    )


__all__ = [
    "TrainedModel",
    "LEAK_PATTERNS",
    "ID_PATTERNS",
    "DATE_PATTERNS",
    "is_safe_feature",
    "split_by_time",
    "train_catboost",
    "train_lgbm",
    "train_logistic_woe",
    "train_weighted_ensemble",
    "train_xgboost",
    "train_stacking_ensemble",
]
