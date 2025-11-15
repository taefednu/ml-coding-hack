from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
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
from src.feature_eng import build_features  # noqa: E402
from src.fairness import group_fairness_report  # noqa: E402
from src.metrics import compute_metrics, report_to_dict  # noqa: E402
from src.modeling import split_by_time, train_catboost, train_lgbm, train_logistic_woe  # noqa: E402
from src.scorecard import ScorecardConfig, export_scorecard, pd_to_score  # noqa: E402
from src.utils import ArtifactPaths, get_logger, load_config, save_json, seed_everything  # noqa: E402

LOGGER = get_logger(__name__)


def _merge_frames(frames: List[pd.DataFrame], key: Optional[str]) -> pd.DataFrame:
    if not frames:
        raise RuntimeError("No frames to merge")
    if key and all(key in frame.columns for frame in frames):
        master = frames[0]
        for fra in frames[1:]:
            master = master.merge(fra, on=key, how="left", suffixes=("", "_dup"))
        return master
    return pd.concat(frames, axis=0, ignore_index=True)


def _detect_column(df: pd.DataFrame, candidates: List[str], fallback: Optional[str] = None) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    if fallback and fallback in df.columns:
        return fallback
    raise ValueError("Failed to detect required column")


def _drop_sensitive(df: pd.DataFrame, sensitive: List[str]) -> pd.DataFrame:
    drop_cols = [col for col in sensitive if col in df.columns]
    if drop_cols:
        LOGGER.info("Dropping sensitive columns from modeling: %s", drop_cols)
    return df.drop(columns=drop_cols, errors="ignore")


def train_pipeline(config: Dict[str, any]) -> None:
    seed_everything(config.get("seed", 42))
    paths = ArtifactPaths(
        models_dir=Path(config["paths"]["models_dir"]),
        artifacts_dir=Path(config["paths"]["artifacts_dir"]),
        reports_dir=Path(config["paths"].get("reports_dir", "artifacts/reports")),
    ).ensure()
    data_dir = Path(config["paths"]["data_dir"])
    inventory = data_loading.scan_data_dir(data_dir)
    frames: List[pd.DataFrame] = []
    for summary in inventory:
        try:
            frames.append(data_loading.read_dataset(summary.path))
        except Exception as exc:
            LOGGER.warning("Skipping %s: %s", summary.path, exc)
    if not frames:
        raise RuntimeError("No readable input files")
    client_id = config["split"].get("client_id_col")
    df = _merge_frames(frames, client_id if client_id and "<<" not in client_id else None)
    target_col = config["target"].get("column")
    if not target_col or "<<" in target_col:
        target_candidates = data_loading.guess_target_columns(df.columns)
        target_col = _detect_column(df, target_candidates)
    date_col = config["split"].get("date_column")
    if not date_col or "<<" in date_col:
        date_candidates = data_loading.guess_date_columns(df.columns)
        date_col = _detect_column(df, date_candidates)
    df = df.dropna(subset=[target_col, date_col])
    feature_cfg = config.get("feature_engineering", {})
    engineered = build_features(df, feature_cfg, date_col=date_col, entity_col=client_id if client_id in df.columns else None)
    df_full = pd.concat([df, engineered], axis=1)
    df_full = df_full.loc[:, df_full.notna().any()]
    sensitive_cols = config.get("fairness", {}).get("sensitive_cols", [])
    modeling_df = _drop_sensitive(df_full, sensitive_cols)
    model_splits = split_by_time(modeling_df, date_col, config["split"]["train_end"], config["split"]["valid_end"])
    raw_splits = split_by_time(df_full, date_col, config["split"]["train_end"], config["split"]["valid_end"])
    train_df, valid_df, oot_df = model_splits["train"], model_splits["valid"], model_splits["oot"]
    oot_raw = raw_splits["oot"]

    champion = train_logistic_woe(train_df, valid_df, target_col, config["modeling"].get("logistic", {}))
    cat_model = train_catboost(
        train_df,
        valid_df,
        target_col,
        config["modeling"].get("catboost", {}),
        date_col,
        client_id if client_id and client_id in modeling_df.columns else None,
        config["modeling"].get("cv_folds", 1),
    )
    lgbm_model = train_lgbm(
        train_df,
        valid_df,
        target_col,
        config["modeling"].get("lightgbm", {}),
        date_col,
        client_id if client_id and client_id in modeling_df.columns else None,
        config["modeling"].get("cv_folds", 1),
    )
    challengers = [m for m in [cat_model, lgbm_model] if m is not None]
    challenger = max(challengers, key=lambda m: m.metrics.get("roc_auc", 0.0)) if challengers else None
    production_model = challenger or champion

    champion_path = paths.models_dir / "champion_model.pkl"
    champion.save(champion_path)
    if challenger:
        challenger_path = paths.models_dir / "challenger_model.pkl"
        challenger.save(challenger_path)
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
