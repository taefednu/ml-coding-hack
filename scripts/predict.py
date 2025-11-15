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


def predict_batch(config: dict, input_path: str, output_path: str) -> Path:
    models_dir = Path(config["paths"]["models_dir"])
    model = joblib.load(models_dir / "best_model.pkl")
    calibrator_path = models_dir / "calibrator.pkl"
    calibrator = ProbabilityCalibrator.load(calibrator_path) if calibrator_path.exists() else None
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
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    predict_batch(config, args.input, args.output)


if __name__ == "__main__":
    main()
