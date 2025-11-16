from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from src import data_loading
from src.calibration import ProbabilityCalibrator
from src.scorecard import ScorecardConfig, pd_to_score
from src.utils import get_logger, load_config

LOGGER = get_logger(__name__)


def predict_batch(config: dict, input_path: str = None, output_path: str = None, use_evaluation: bool = False) -> Path:
    """Predict on batch data.
    
    Args:
        config: Configuration dictionary
        input_path: Path to input CSV/Parquet file (if None and use_evaluation=True, uses evaluation_dir)
        output_path: Path to output CSV file (if None, uses default)
        use_evaluation: If True, load master dataset from evaluation_dir and build features
    """
    models_dir = Path(config["paths"]["models_dir"])
    model = joblib.load(models_dir / "best_model.pkl")
    calibrator_path = models_dir / "calibrator.pkl"
    calibrator = ProbabilityCalibrator.load(calibrator_path) if calibrator_path.exists() else None
    
    if use_evaluation and input_path is None:
        # Load evaluation dataset and process exactly like training data
        from src.feature_eng import build_features, build_smart_interactions
        from src.modeling import split_by_time, is_safe_feature
        from scripts.train import _augment_with_features, _select_feature_columns, _align_frames, _drop_sensitive
        
        LOGGER.info("Loading evaluation dataset from %s", config["paths"].get("evaluation_dir"))
        master_df, credit_history = data_loading.load_master_dataset(config, use_evaluation=True)
        
        # Process exactly like training pipeline
        target_col = config["target"].get("column")
        date_col = config["split"].get("date_column", "application_date")
        id_col = config.get("merging", {}).get("id_col", "customer_ref")
        app_col = config.get("split", {}).get("application_id_col", "application_id")
        client_id = config["split"].get("client_id_col")
        
        # Build credit history features
        if credit_history is not None and not credit_history.empty:
            credit_features = data_loading.build_credit_history_features(credit_history, master_df, config)
            if not credit_features.empty:
                master_df = master_df.merge(credit_features, on=app_col, how="left")
        
        # Note: We don't dropna on target_col for evaluation (it might not exist)
        # But we should dropna on date_col if it exists
        if date_col in master_df.columns:
            master_df = master_df.dropna(subset=[date_col])
            master_df = master_df.sort_values(date_col)
        
        # Build features using same function as training
        feature_cfg = config.get("feature_engineering", {})
        augmented_df = _augment_with_features(master_df.copy(), feature_cfg, date_col, client_id)
        
        # Select feature columns (same logic as training)
        id_like_cols = config.get("merging", {}).get("id_like_cols", [])
        forbidden_ids = [col for col in [id_col, app_col] if col]
        forbidden_ids.extend([col for col in id_like_cols if col])
        forbidden_ids = list(dict.fromkeys(forbidden_ids))
        
        # Use same column selection logic as training
        # _select_feature_columns expects a dict of frames, so we simulate it
        temp_frames = {"eval": augmented_df}
        safe_columns = _select_feature_columns(
            temp_frames,
            target_col or "default",
            date_col,
            forbidden_ids,
        )
        
        # Align columns (same as training)
        aligned_frames = _align_frames(temp_frames, safe_columns)
        augmented_df = aligned_frames["eval"]
        
        # Drop sensitive columns
        sensitive_cols = config.get("fairness", {}).get("sensitive_cols", [])
        augmented_df = _drop_sensitive(augmented_df, sensitive_cols)
        
        # Build smart interactions if enabled (with same parameters as training)
        interactions_cfg = feature_cfg.get("interactions", {})
        if interactions_cfg.get("enabled", False):
            from pathlib import Path
            artifacts_dir = Path(config["paths"]["artifacts_dir"])
            feature_strength_path = artifacts_dir / "feature_strength.json"
            top_k = interactions_cfg.get("top_k", 12)
            max_pairs = interactions_cfg.get("max_pairs", 20)
            
            # Load diagnostics if available
            suspicious_features = []
            if feature_strength_path.exists():
                import json
                try:
                    with open(feature_strength_path) as f:
                        diag_data = json.load(f)
                        suspicious_features = diag_data.get("suspicious", [])
                except:
                    pass
            
            min_spearman = interactions_cfg.get("min_spearman_correlation")
            if min_spearman is None:
                min_spearman = config.get("feature_selection", {}).get("min_spearman_correlation", 0.01)
            
            smart_interactions = build_smart_interactions(
                augmented_df,
                feature_strength_path=str(feature_strength_path) if feature_strength_path.exists() else None,
                top_k=top_k,
                max_pairs=max_pairs,
                forbidden_ids=forbidden_ids,
                suspicious_features=suspicious_features,
                min_spearman_corr=float(min_spearman),
            )
            if not smart_interactions.empty:
                augmented_df = pd.concat([augmented_df, smart_interactions], axis=1)
                LOGGER.info("Added %d smart interaction features to evaluation set", smart_interactions.shape[1])
        
        # Select only feature columns that model expects
        # Model.predict_proba will handle preprocessing if it's a Pipeline
        df = augmented_df.copy()
        for col in model.feature_cols:
            if col not in df.columns:
                df[col] = 0
        # Ensure columns are in the same order as model expects
        df = df[model.feature_cols]
    else:
        # Load from file
        if input_path is None:
            raise ValueError("input_path must be provided if use_evaluation=False")
        df = data_loading.read_dataset(input_path)
        for col in model.feature_cols:
            if col not in df.columns:
                df[col] = 0
        df = df[model.feature_cols]
    
    preds = model.predict_proba(df)
    if calibrator:
        preds = calibrator.transform(preds)
    score_cfg = ScorecardConfig(**config.get("scorecard", {}))
    scores = pd.to_numeric(pd.Series(pd_to_score(preds, score_cfg)))
    
    if output_path is None:
        output_path = str(Path(config["paths"]["artifacts_dir"]) / "evaluation_predictions.csv")
    output = Path(output_path)
    payload = df.copy()
    payload["pd"] = preds
    payload["score"] = scores
    payload.to_csv(output, index=False)
    LOGGER.info("Scored %d rows. Output saved to %s", payload.shape[0], output)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch scoring script")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--input", type=str, help="Input CSV/Parquet file (optional if --evaluation is used)")
    parser.add_argument("--output", type=str, help="Output CSV file (default: artifacts/evaluation_predictions.csv)")
    parser.add_argument("--evaluation", action="store_true", help="Use evaluation_set directory for prediction")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.evaluation and args.input:
        LOGGER.warning("Both --evaluation and --input provided. Using --evaluation mode.")
    predict_batch(config, args.input, args.output, use_evaluation=args.evaluation)


if __name__ == "__main__":
    main()
