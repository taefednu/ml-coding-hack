from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import data_loading  # noqa: E402
from src.calibration import ProbabilityCalibrator  # noqa: E402
from src.fairness import group_fairness_report  # noqa: E402
from src.metrics import compute_metrics, lift_curve, reliability_curve, report_to_dict  # noqa: E402
from src.utils import get_logger, load_config, save_json  # noqa: E402

LOGGER = get_logger(__name__)


def _resolve_target(df: pd.DataFrame, config: dict) -> str:
    target_col = config["target"].get("column")
    if target_col and "<<" not in target_col and target_col in df.columns:
        return target_col
    candidates = data_loading.guess_target_columns(df.columns)
    if not candidates:
        raise ValueError("Unable to detect target column in OOT predictions")
    return candidates[0]


def evaluate(config: dict) -> Path:
    artifacts_dir = Path(config["paths"]["artifacts_dir"])
    models_dir = Path(config["paths"]["models_dir"])
    oot_path = artifacts_dir / "oot_predictions.csv"
    if not oot_path.exists():
        raise FileNotFoundError("Run training first to generate oot_predictions.csv")
    df = pd.read_csv(oot_path)
    target_col = _resolve_target(df, config)
    if target_col not in df.columns:
        raise ValueError(f"Target column {target_col} not found in OOT predictions")
    model = joblib.load(models_dir / "best_model.pkl")
    calibrator_path = models_dir / "calibrator.pkl"
    calibrator = ProbabilityCalibrator.load(calibrator_path) if calibrator_path.exists() else None
    if "pd" not in df.columns:
        preds = model.predict_proba(df[model.feature_cols])
        if calibrator:
            preds = calibrator.transform(preds)
        df["pd"] = preds
    metrics = compute_metrics(df[target_col].values, df["pd"].values)
    report = report_to_dict(metrics)
    reliability = reliability_curve(df[target_col].values, df["pd"].values)
    lift = lift_curve(df[target_col].values, df["pd"].values)
    fairness_cols = [col for col in config.get("fairness", {}).get("sensitive_cols", []) if col in df.columns]
    fairness = {}
    if fairness_cols:
        fairness = group_fairness_report(
            df,
            fairness_cols,
            target_col,
            pred_col="pd",
            score_col="score" if "score" in df.columns else None,
            min_group_size=config.get("fairness", {}).get("min_group_size", 50),
        )
        save_json(fairness, artifacts_dir / "fairness_report.json")
    else:
        LOGGER.info("No fairness columns configured")
    report_path = artifacts_dir / "evaluation_report.json"
    payload = {
        "metrics": report,
        "reliability": reliability,
        "lift": lift,
        "fairness": fairness,
    }
    save_json(payload, report_path)
    LOGGER.info("Evaluation report saved to %s", report_path)
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PD model on OOT")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    evaluate(config)


if __name__ == "__main__":
    main()
