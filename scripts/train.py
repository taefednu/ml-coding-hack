from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
JOBLIB_TMP = ROOT / "tmp" / "joblib"
JOBLIB_TMP.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("JOBLIB_TEMP_FOLDER", str(JOBLIB_TMP))
os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")
warnings.filterwarnings("ignore", message=".*joblib will operate in serial mode.*")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import data_loading  # noqa: E402
from src.calibration import ProbabilityCalibrator  # noqa: E402
from src.drift import psi_report  # noqa: E402
from src.explainability import (  # noqa: E402
    challenger_shap_summary,
    champion_coefficients,
    export_champion_explainability,
)
from src.feature_eng import build_features, build_smart_interactions  # noqa: E402
from src.fairness import group_fairness_report  # noqa: E402
from src.metrics import compute_metrics, report_to_dict  # noqa: E402
from src.modeling import (  # noqa: E402
    is_safe_feature,
    split_by_time,
    train_catboost,
    train_lgbm,
    train_logistic_woe,
    train_weighted_ensemble,
    train_xgboost,
    train_stacking_ensemble,
)
from src.scorecard import ScorecardConfig, export_scorecard, pd_to_score  # noqa: E402
from src.utils import ArtifactPaths, get_logger, load_config, save_json, seed_everything  # noqa: E402

LOGGER = get_logger(__name__)


def _drop_sensitive(df: pd.DataFrame, sensitive: List[str]) -> pd.DataFrame:
    drop_cols = [col for col in sensitive if col in df.columns]
    if drop_cols:
        LOGGER.info("Dropping sensitive columns from modeling: %s", drop_cols)
    return df.drop(columns=drop_cols, errors="ignore")


def _augment_with_features(
    df_slice: pd.DataFrame,
    feature_cfg: Dict[str, any],
    date_col: str,
    entity_col: Optional[str],
) -> pd.DataFrame:
    if df_slice.empty:
        return df_slice.copy()
    entity = entity_col if entity_col and entity_col in df_slice.columns else None
    enable_default_detection = feature_cfg.get("enable_default_detection", False)
    engineered = build_features(
        df_slice, feature_cfg, date_col=date_col, entity_col=entity,
        enable_default_detection=enable_default_detection
    )
    augmented = pd.concat([df_slice, engineered], axis=1)
    augmented = augmented.loc[:, augmented.notna().any()]
    return augmented


def _select_feature_columns(
    frames: Dict[str, pd.DataFrame],
    target_col: str,
    date_col: str,
    id_cols: List[str],
) -> List[str]:
    union: set[str] = set()
    for frame in frames.values():
        union.update(frame.columns)
    ordered = sorted(union)
    keep: List[str] = []
    for col in ordered:
        if col == target_col or col == date_col:
            keep.append(col)
        elif is_safe_feature(col, target_col, id_cols):
            keep.append(col)
    return keep


def _align_frames(frames: Dict[str, pd.DataFrame], columns: List[str]) -> Dict[str, pd.DataFrame]:
    aligned: Dict[str, pd.DataFrame] = {}
    for name, frame in frames.items():
        frame = frame.copy()
        missing = [col for col in columns if col not in frame.columns]
        if missing:
            for col in missing:
                frame[col] = np.nan
        aligned[name] = frame.loc[:, columns]
    return aligned


def _encode_feature(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.nunique(dropna=True) >= 2:
        return numeric
    if series.dtype == "bool":
        return series.astype(float)
    cat = series.astype(str)
    codes, _ = pd.factorize(cat, sort=True)
    encoded = pd.Series(codes.astype(float), index=series.index)
    encoded = encoded.where(series.notna(), np.nan)
    return encoded


def _single_feature_auc(series: pd.Series, target: pd.Series) -> float:
    if series is None or target is None:
        return 0.5
    encoded = _encode_feature(series)
    mask = encoded.notna() & target.notna()
    if mask.sum() < 20 or target[mask].nunique() < 2 or encoded[mask].nunique() < 2:
        return 0.5
    ranked = encoded[mask].rank(pct=True, method="average")
    try:
        return float(roc_auc_score(target[mask], ranked))
    except ValueError:
        return 0.5


def _deterministic_mapping(series: pd.Series, target: pd.Series) -> bool:
    mask = series.notna() & target.notna()
    if mask.sum() == 0:
        return False
    grouped = (
        pd.DataFrame({"feature": series[mask], "target": target[mask]})
        .groupby("feature")["target"]
        .nunique()
    )
    return bool(len(grouped) > 0 and grouped.max() == 1)


def _select_features_by_shap(
    models: List[Any],
    valid_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    top_k: int = 100,
    max_samples: int = 500,
) -> List[str]:
    """Select top features based on SHAP importance from ensemble of models."""
    try:
        import shap
    except ImportError:
        LOGGER.warning("SHAP not available, skipping SHAP-based feature selection")
        return feature_cols
    
    available_models = [m for m in models if m is not None and hasattr(m, "estimator")]
    if not available_models:
        return feature_cols
    
    # Sample data for SHAP computation
    sample_size = min(max_samples, len(valid_df))
    sample_df = valid_df.sample(n=sample_size, random_state=42) if sample_size < len(valid_df) else valid_df
    X_sample = sample_df[feature_cols]
    
    shap_importances = {}
    for model in available_models:
        try:
            estimator = model.estimator
            if hasattr(estimator, "steps"):  # Pipeline
                prep = estimator.named_steps.get("prep")
                model_est = estimator.named_steps.get("model")
                if prep and model_est:
                    X_prep = prep.transform(X_sample)
                    explainer = shap.TreeExplainer(model_est)
                    shap_values = explainer.shap_values(X_prep)
                    if isinstance(shap_values, list):
                        shap_values = shap_values[1]  # Positive class
                    importance = np.abs(shap_values).mean(0)
                    feature_names = prep.get_feature_names_out()
                    for i, name in enumerate(feature_names):
                        orig_name = name.split("__")[-1] if "__" in name else name
                        shap_importances[orig_name] = shap_importances.get(orig_name, 0) + importance[i]
            elif hasattr(estimator, "predict_proba"):
                explainer = shap.TreeExplainer(estimator)
                shap_values = explainer.shap_values(X_sample)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                importance = np.abs(shap_values).mean(0)
                for i, col in enumerate(feature_cols):
                    shap_importances[col] = shap_importances.get(col, 0) + importance[i]
        except Exception as e:
            LOGGER.warning("SHAP computation failed for model %s: %s", model.name if hasattr(model, "name") else "unknown", e)
            continue
    
    if not shap_importances:
        return feature_cols
    
    # Select top K features
    sorted_features = sorted(shap_importances.items(), key=lambda x: x[1], reverse=True)
    top_features = [feat for feat, _ in sorted_features[:top_k]]
    LOGGER.info("SHAP-based feature selection: selected %d top features", len(top_features))
    return top_features


def _ensemble_feature_importance(
    models: List[Any],
    feature_cols: List[str],
) -> Dict[str, float]:
    """Compute average feature importance across multiple tree-based models."""
    importances = {}
    count = 0
    
    for model in models:
        if model is None:
            continue
        try:
            estimator = model.estimator
            if hasattr(estimator, "steps"):  # Pipeline
                model_est = estimator.named_steps.get("model")
                prep = estimator.named_steps.get("prep")
                if model_est and hasattr(model_est, "feature_importances_"):
                    feat_names = prep.get_feature_names_out() if prep else feature_cols
                    for i, name in enumerate(feat_names):
                        orig_name = name.split("__")[-1] if "__" in name else name
                        if orig_name in feature_cols:
                            importances[orig_name] = importances.get(orig_name, 0) + model_est.feature_importances_[i]
                            count += 1
            elif hasattr(estimator, "feature_importances_"):
                for i, col in enumerate(feature_cols):
                    if i < len(estimator.feature_importances_):
                        importances[col] = importances.get(col, 0) + estimator.feature_importances_[i]
                        count += 1
        except Exception:
            continue
    
    if count > 0:
        return {k: v / count for k, v in importances.items()}
    return {}


def _run_feature_diagnostics(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    target_col: str,
    date_col: str,
    artifacts_dir: Path,
    cfg: Dict[str, any],
) -> Dict[str, List[str]]:
    sel_cfg = cfg or {}
    suspicious_auc = float(sel_cfg.get("suspicious_auc", 0.98))
    weak_eps = float(sel_cfg.get("weak_auc_tolerance", 0.02))
    low_var_ratio = float(sel_cfg.get("low_variance_ratio", 0.001))
    drop_weak = bool(sel_cfg.get("drop_weak", False))
    min_spearman_corr = float(sel_cfg.get("min_spearman_correlation", 0.01))
    top_k = int(sel_cfg.get("report_top_k", 20))
    features = [col for col in train_df.columns if col not in {target_col, date_col}]
    stats: List[Dict[str, float]] = []
    low_variance: set[str] = set()
    weak: set[str] = set()
    weak_spearman: set[str] = set()
    suspicious: set[str] = set()
    y_train = train_df[target_col].astype(float)
    y_valid = valid_df[target_col].astype(float) if target_col in valid_df.columns else None
    for col in features:
        train_series = train_df[col]
        auc_train = _single_feature_auc(train_series, y_train)
        auc_valid = _single_feature_auc(valid_df[col], y_valid) if y_valid is not None and col in valid_df.columns else None
        encoded = _encode_feature(train_series)
        corr = 0.0
        spearman_corr = 0.0
        encoded_std = float(encoded.std(skipna=True)) if encoded.notna().sum() > 1 else 0.0
        target_std = float(y_train.std()) if y_train.notna().sum() > 1 else 0.0
        if encoded_std > 0 and target_std > 0:
            try:
                corr = float(encoded.corr(y_train))  # Pearson
            except Exception:
                corr = 0.0
            # Spearman correlation (monotonic relationship)
            try:
                mask = encoded.notna() & y_train.notna()
                if mask.sum() >= 10:  # Need at least 10 valid pairs
                    spearman_result = spearmanr(encoded[mask], y_train[mask])
                    if not np.isnan(spearman_result.correlation):
                        spearman_corr = float(spearman_result.correlation)
            except Exception:
                spearman_corr = 0.0
        unique_ratio = train_series.nunique(dropna=True) / max(len(train_series), 1)
        stats.append(
            {
                "feature": col,
                "auc_train": auc_train,
                "auc_valid": auc_valid,
                "corr": corr,
                "spearman_corr": spearman_corr,
                "unique_ratio": float(unique_ratio),
            }
        )
        if unique_ratio <= low_var_ratio or encoded.std(skipna=True) == 0:
            low_variance.add(col)
        if (auc_train >= suspicious_auc) or (auc_valid is not None and auc_valid >= suspicious_auc):
            suspicious.add(col)
        if _deterministic_mapping(train_series, y_train):
            suspicious.add(col)
        if abs(auc_train - 0.5) <= weak_eps and (auc_valid is None or abs(auc_valid - 0.5) <= weak_eps):
            weak.add(col)
        # Filter features with weak Spearman correlation
        if abs(spearman_corr) < min_spearman_corr:
            weak_spearman.add(col)
    mi_scores: Dict[str, float] = {}
    mi_sample = train_df
    sample_size = int(sel_cfg.get("mutual_info_sample", 0))
    if sample_size and len(mi_sample) > sample_size:
        mi_sample = mi_sample.sample(sample_size, random_state=42)
    if features and not mi_sample.empty:
        encoded_matrix: Dict[str, pd.Series] = {}
        for col in features:
            encoded = _encode_feature(mi_sample[col])
            fill_value = encoded.median()
            if np.isnan(fill_value):
                fill_value = 0.0
            encoded_matrix[col] = encoded.fillna(fill_value)
        feature_matrix = pd.DataFrame(encoded_matrix, index=mi_sample.index)
        if not feature_matrix.empty:
            mi_values = mutual_info_classif(
                feature_matrix.values,
                mi_sample[target_col].values,
                random_state=42,
            )
            mi_scores = {col: float(score) for col, score in zip(feature_matrix.columns, mi_values)}
    # Top features by Spearman correlation
    top_spearman = sorted(
        [{"feature": s["feature"], "spearman_corr": s["spearman_corr"]} for s in stats],
        key=lambda x: abs(x["spearman_corr"]),
        reverse=True,
    )[:top_k]
    
    payload = {
        "top_auc": sorted(stats, key=lambda x: x["auc_train"], reverse=True)[:top_k],
        "top_spearman": top_spearman,
        "top_mutual_info": sorted(
            [{"feature": k, "mutual_info": v} for k, v in mi_scores.items()],
            key=lambda x: x["mutual_info"],
            reverse=True,
        )[:top_k],
        "low_variance": sorted(low_variance),
        "weak_auc": sorted(weak),
        "weak_spearman": sorted(weak_spearman),
        "suspicious": sorted(suspicious),
    }
    save_json(payload, artifacts_dir / "feature_strength.json")
    
    # Log top Spearman correlations
    if top_spearman:
        LOGGER.info("Top 10 features by Spearman correlation (absolute value):")
        for feat_info in top_spearman[:10]:
            LOGGER.info("  %s: %.4f", feat_info["feature"], feat_info["spearman_corr"])
    
    if weak_spearman:
        LOGGER.info("Found %d features with weak Spearman correlation (< %.3f): %s", 
                   len(weak_spearman), min_spearman_corr, sorted(weak_spearman)[:20])
    
    drop_columns = set(low_variance) | set(suspicious)
    if drop_weak:
        drop_columns |= weak
    if sel_cfg.get("drop_weak_spearman", False):
        drop_columns |= weak_spearman
        LOGGER.info("Dropping %d features with weak Spearman correlation", len(weak_spearman))
    return {"drop_columns": sorted(drop_columns), "suspicious": sorted(suspicious)}


def train_pipeline(config: Dict[str, any]) -> None:
    seed_everything(config.get("seed", 42))
    paths = ArtifactPaths(
        models_dir=Path(config["paths"]["models_dir"]),
        artifacts_dir=Path(config["paths"]["artifacts_dir"]),
        reports_dir=Path(config["paths"].get("reports_dir", "artifacts/reports")),
    ).ensure()
    master_df, credit_history = data_loading.load_master_dataset(config)
    target_col = config["target"].get("column")
    date_col = config["split"].get("date_column")
    id_col = config.get("merging", {}).get("id_col", "customer_ref")
    app_col = config.get("split", {}).get("application_id_col", "application_id")
    client_id = config["split"].get("client_id_col")
    if credit_history is not None and not credit_history.empty:
        credit_features = data_loading.build_credit_history_features(credit_history, master_df, config)
        if not credit_features.empty:
            master_df = master_df.merge(credit_features, on=app_col, how="left")
    master_df = master_df.dropna(subset=[target_col, date_col])
    master_df = master_df.sort_values(date_col)
    feature_cfg = config.get("feature_engineering", {})
    raw_splits = split_by_time(master_df, date_col, config["split"]["train_end"], config["split"]["valid_end"])
    augmented_splits = {
        name: _augment_with_features(split.copy(), feature_cfg, date_col, client_id)
        for name, split in raw_splits.items()
    }
    id_like_cols = config.get("merging", {}).get("id_like_cols", [])
    forbidden_ids = [col for col in [id_col, app_col] if col]
    forbidden_ids.extend([col for col in id_like_cols if col])
    forbidden_ids = list(dict.fromkeys(forbidden_ids))
    safe_columns = _select_feature_columns(
        augmented_splits,
        target_col,
        date_col,
        forbidden_ids,
    )
    augmented_splits = _align_frames(augmented_splits, safe_columns)
    diag = _run_feature_diagnostics(
        augmented_splits["train"],
        augmented_splits["valid"],
        target_col,
        date_col,
        paths.artifacts_dir,
        config.get("feature_selection", {}),
    )
    if diag.get("suspicious"):
        LOGGER.warning(
            "Single-feature leakage suspects detected: %s",
            diag["suspicious"][:20],
        )
    drop_candidates = diag.get("drop_columns", [])
    if drop_candidates:
        LOGGER.info("Dropping %d low-signal/leakage features: %s", len(drop_candidates), drop_candidates[:20])
        for name in augmented_splits:
            augmented_splits[name] = augmented_splits[name].drop(columns=drop_candidates, errors="ignore")
    sensitive_cols = config.get("fairness", {}).get("sensitive_cols", [])
    for name in augmented_splits:
        augmented_splits[name] = _drop_sensitive(augmented_splits[name], sensitive_cols)
    
    # Smart interactions based on top features
    interactions_cfg = config.get("feature_engineering", {}).get("interactions", {})
    if interactions_cfg.get("enabled", False):
        feature_strength_path = paths.artifacts_dir / "feature_strength.json"
        top_k = interactions_cfg.get("top_k", 12)
        max_pairs = interactions_cfg.get("max_pairs", 20)
        suspicious_features = diag.get("suspicious", [])
        id_like_cols = config.get("merging", {}).get("id_like_cols", [])
        forbidden_ids = [col for col in [id_col, app_col] if col]
        forbidden_ids.extend(id_like_cols)
        
        min_spearman = interactions_cfg.get("min_spearman_correlation")
        if min_spearman is None:
            # Use default from feature_selection config
            min_spearman = config.get("feature_selection", {}).get("min_spearman_correlation", 0.01)
        
        for name, split_df in augmented_splits.items():
            smart_interactions = build_smart_interactions(
                split_df,
                feature_strength_path=str(feature_strength_path) if feature_strength_path.exists() else None,
                top_k=top_k,
                max_pairs=max_pairs,
                forbidden_ids=forbidden_ids,
                suspicious_features=suspicious_features,
                min_spearman_corr=float(min_spearman),
            )
            if not smart_interactions.empty:
                augmented_splits[name] = pd.concat([split_df, smart_interactions], axis=1)
                LOGGER.info("Added %d smart interaction features to %s", smart_interactions.shape[1], name)
    train_df, valid_df, oot_df = (
        augmented_splits["train"],
        augmented_splits["valid"],
        augmented_splits["oot"],
    )
    oot_raw = raw_splits["oot"]

    champion = train_logistic_woe(train_df, valid_df, target_col, config["modeling"].get("logistic", {}))
    cat_model = train_catboost(
        train_df,
        valid_df,
        target_col,
        config["modeling"].get("catboost", {}),
        date_col,
        client_id if client_id and client_id in train_df.columns else None,
        config["modeling"].get("cv_folds", 1),
    )
    lgbm_model = train_lgbm(
        train_df,
        valid_df,
        target_col,
        config["modeling"].get("lightgbm", {}),
        date_col,
        client_id if client_id and client_id in train_df.columns else None,
        config["modeling"].get("cv_folds", 1),
    )
    xgb_model = train_xgboost(
        train_df,
        valid_df,
        target_col,
        config["modeling"].get("xgboost", {}),
        date_col,
        client_id if client_id and client_id in train_df.columns else None,
        config["modeling"].get("cv_folds", 1),
    )
    weighted_ensemble = train_weighted_ensemble(
        [champion, cat_model, lgbm_model, xgb_model],
        train_df,
        valid_df,
        target_col,
        config["modeling"].get("ensemble", {}),
    )
    stacking_ensemble = train_stacking_ensemble(
        [cat_model, lgbm_model, xgb_model],
        train_df,
        valid_df,
        target_col,
        config["modeling"].get("stacking", {}),
    )
    challengers = [m for m in [cat_model, lgbm_model, xgb_model, weighted_ensemble, stacking_ensemble] if m is not None]
    tree_challengers = [m for m in [cat_model, lgbm_model, xgb_model] if m is not None]
    ensemble_model = stacking_ensemble if stacking_ensemble else weighted_ensemble
    challenger = max(tree_challengers, key=lambda m: m.metrics.get("roc_auc", 0.0)) if tree_challengers else None
    
    # Get feature columns from models or dataframe
    feature_cols = [col for col in train_df.columns if col not in {target_col, date_col}]
    if tree_challengers and tree_challengers[0].feature_cols:
        feature_cols = tree_challengers[0].feature_cols
    
    # SHAP-based feature selection if enabled
    sel_cfg = config.get("feature_selection", {})
    if sel_cfg.get("use_shap_selection", False) and tree_challengers:
        top_k = sel_cfg.get("shap_top_k", 100)
        selected_features = _select_features_by_shap(
            tree_challengers,
            valid_df,
            feature_cols,
            target_col,
            top_k=top_k,
        )
        if len(selected_features) < len(feature_cols):
            LOGGER.info("SHAP feature selection: reducing from %d to %d features", len(feature_cols), len(selected_features))
            feature_cols = selected_features
            train_df = train_df[[target_col, date_col] + [f for f in feature_cols if f in train_df.columns]]
            valid_df = valid_df[[target_col, date_col] + [f for f in feature_cols if f in valid_df.columns]]
            oot_df = oot_df[[target_col, date_col] + [f for f in feature_cols if f in oot_df.columns]]
    
    # Compute ensemble feature importance
    if tree_challengers:
        ensemble_importance = _ensemble_feature_importance(tree_challengers, feature_cols)
        if ensemble_importance:
            top_features = sorted(ensemble_importance.items(), key=lambda x: x[1], reverse=True)[:20]
            LOGGER.info("Top 20 ensemble feature importances:")
            for feat, imp in top_features:
                LOGGER.info("  %s: %.6f", feat, imp)
    production_model = champion
    current_auc = champion.metrics.get("roc_auc", 0.0)
    for candidate in challengers:
        cand_auc = candidate.metrics.get("roc_auc", 0.0)
        suspicious_flag = candidate.extras.get("is_suspicious") if isinstance(candidate.extras, dict) else False
        if suspicious_flag:
            LOGGER.warning("Candidate %s flagged as suspicious; skipping", candidate.name)
            continue
        if cand_auc >= 0.99:
            LOGGER.warning("Candidate %s achieved suspicious ROC-AUC=%.4f; skipping to avoid leakage", candidate.name, cand_auc)
            continue
        if cand_auc > current_auc + 0.005:
            LOGGER.info("Candidate %s outperformed current production (%.4f vs %.4f)", candidate.name, cand_auc, current_auc)
            production_model = candidate
            current_auc = cand_auc
        else:
            LOGGER.info("Candidate %s did not outperform production (%.4f vs %.4f)", candidate.name, cand_auc, current_auc)

    champion_path = paths.models_dir / "champion_model.pkl"
    champion.save(champion_path)
    if challenger:
        challenger_path = paths.models_dir / "challenger_model.pkl"
        challenger.save(challenger_path)
    if ensemble_model:
        ensemble_path = paths.models_dir / "ensemble_model.pkl"
        ensemble_model.save(ensemble_path)
    if xgb_model:
        xgb_path = paths.models_dir / "xgb_model.pkl"
        xgb_model.save(xgb_path)
    if stacking_ensemble:
        stacking_path = paths.models_dir / "stacking_ensemble.pkl"
        stacking_ensemble.save(stacking_path)
    production_path = paths.models_dir / "best_model.pkl"
    production_model.save(production_path)
    LOGGER.info("Production model: %s", production_model.name)

    calibrator_metrics = {}
    calibrator_path = paths.models_dir / "calibrator.pkl"
    calibrator = None
    if config.get("calibration", {}).get("enabled", True):
        calibrator = ProbabilityCalibrator()
        prod_valid_pred = production_model.predict_proba(valid_df)
        calibration_result = calibrator.fit(valid_df[target_col].values, prod_valid_pred)
        calibrator_metrics = calibration_result.metrics
        calibrator.save(calibrator_path)
    score_cfg = ScorecardConfig(**config.get("scorecard", {}))

    def _apply_model(df_slice: pd.DataFrame) -> pd.Series:
        preds = production_model.predict_proba(df_slice)
        if calibrator:
            preds = calibrator.transform(preds)
        return pd.Series(preds, index=df_slice.index)

    valid_proba = _apply_model(valid_df)
    oot_proba = _apply_model(oot_df)
    oot_scores = pd_to_score(oot_proba.values, score_cfg)

    metrics_payload = {
        "champion": champion.metrics,
        "challenger": challenger.metrics if challenger else None,
        "ensemble": ensemble_model.metrics if ensemble_model else None,
        "calibration": calibrator_metrics,
        "valid": report_to_dict(compute_metrics(valid_df[target_col].values, valid_proba.values)),
        "oot": report_to_dict(compute_metrics(oot_df[target_col].values, oot_proba.values)),
    }
    drift_payload = psi_report(train_df, valid_df, oot_df)
    save_json(metrics_payload, paths.artifacts_dir / "metrics.json")
    save_json(drift_payload, paths.artifacts_dir / "psi.json")

    oot_with_predictions = oot_raw.copy()
    oot_with_predictions["pd"] = oot_proba.values
    oot_with_predictions["score"] = oot_scores
    oot_with_predictions.to_csv(paths.artifacts_dir / "oot_predictions.csv", index=False)

    scorecard_path = paths.models_dir / "scorecard.yaml"
    champion_bins = []
    woe = champion.extras.get("woe")
    if woe:
        champion_bins = [vars(summary) for summary in woe.get_bin_summaries()]
    coefficients = champion_coefficients(champion)
    champion_points = {row["feature"]: row["coefficient"] * score_cfg.factor for row in coefficients}
    export_scorecard(champion_points, champion_bins, scorecard_path, cfg=score_cfg)

    explain_cfg = config.get("explainability", {})
    export_champion_explainability(
        champion,
        valid_df,
        explain_cfg,
        paths.artifacts_dir / "champion_explainability.json",
    )
    if challenger:
        challenger_shap_summary(
            challenger,
            valid_df,
            explain_cfg,
            paths.artifacts_dir / "challenger_shap.json",
        )

    fairness_cols = [col for col in sensitive_cols if col in oot_with_predictions.columns]
    if fairness_cols:
        fairness_payload = group_fairness_report(
            oot_with_predictions,
            fairness_cols,
            target_col,
            pred_col="pd",
            score_col="score",
            min_group_size=config.get("fairness", {}).get("min_group_size", 50),
        )
        save_json(fairness_payload, paths.artifacts_dir / "fairness_report.json")

    LOGGER.info("Artifacts written to %s", paths.artifacts_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PD model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    train_pipeline(config)


if __name__ == "__main__":
    main()
