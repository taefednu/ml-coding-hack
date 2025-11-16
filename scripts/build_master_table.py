"""CLI script to assemble the master scoring table."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fe_pipeline import preprocess_and_generate_features
from src.merging import MergeConfig
from src.preprocessing import PreprocessingConfig
from src.utils import get_logger, load_config
from src.data_loading import _read_xlsx_naive

LOGGER = get_logger(__name__)


def _detect_parquet_engine() -> bool:
    try:
        import pyarrow  # type: ignore  # noqa: F401

        return True
    except Exception:
        try:
            import fastparquet  # type: ignore  # noqa: F401

            return True
        except Exception:
            return False


PARQUET_ENGINE_AVAILABLE = _detect_parquet_engine()


def _read_excel_with_fallback(path: Path) -> pd.DataFrame:
    try:
        return pd.read_excel(path)
    except ImportError:
        LOGGER.info("openpyxl missing, using naive XLSX reader for %s", path)
        return _read_xlsx_naive(path)


def _load_sources(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    application = pd.read_csv(data_dir / "application_metadata.csv")
    demographics = pd.read_csv(data_dir / "demographics.csv")
    loan_details = _read_excel_with_fallback(data_dir / "loan_details.xlsx")
    financial_ratios = pd.read_json(data_dir / "financial_ratios.jsonl", lines=True)
    if PARQUET_ENGINE_AVAILABLE:
        credit_history = pd.read_parquet(data_dir / "credit_history.parquet")
    else:
        LOGGER.info("Parquet engine missing; credit history aggregates will be skipped.")
        credit_history = pd.DataFrame()
    return application, demographics, loan_details, financial_ratios, credit_history


def _write_output(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    else:
        if PARQUET_ENGINE_AVAILABLE:
            df.to_parquet(path, index=False)
        else:
            fallback = path.with_suffix(".csv")
            LOGGER.info("Parquet engine missing; writing CSV fallback to %s", fallback)
            df.to_csv(fallback, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build master scoring table with advanced FE")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/train"),
        help="Directory containing the raw source files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/master_table.parquet"),
        help="Where to persist the processed dataset (CSV or Parquet).",
    )
    parser.add_argument(
        "--categorical-encoding",
        default="frequency",
        choices=["frequency", "target", "onehot"],
        help="Encoding strategy for categorical columns.",
    )
    parser.add_argument(
        "--enable-woe",
        action="store_true",
        help="Generate monotonic WOE features for champion models.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to configuration file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    LOGGER.info("Loading raw sources from %s", args.data_dir)
    application, demographics, loan_details, ratios, history = _load_sources(args.data_dir)

    # Load config for feature engineering settings
    config = load_config(args.config) if args.config.exists() else {}
    feature_engineering_config = config.get("feature_engineering", {})
    
    # Build merge config from config file or defaults
    merging_cfg = config.get("merging", {})
    merge_cfg = MergeConfig(
        id_col=merging_cfg.get("id_col", "customer_ref"),
        application_id_col=merging_cfg.get("application_id_col", "application_id"),
        credit_history_date_col=merging_cfg.get("credit_history_date_col"),
        dpd_thresholds=tuple(merging_cfg.get("dpd_thresholds", [30, 60, 90])),
    )
    
    prep_cfg = PreprocessingConfig(
        categorical_encoding=args.categorical_encoding,
        enable_woe=args.enable_woe,
    )
    dataset, artifacts = preprocess_and_generate_features(
        application,
        demographics_df=demographics,
        loan_df=loan_details,
        ratios_df=ratios,
        history_df=history,
        merge_config=merge_cfg,
        preprocessing_config=prep_cfg,
        feature_engineering_config=feature_engineering_config,
        return_artifacts=True,
    )

    _write_output(dataset, args.output)
    LOGGER.info(
        "Master table saved to %s (%d rows, %d columns)",
        args.output,
        len(dataset),
        dataset.shape[1],
    )
    LOGGER.info("Merge coverage report: %s", artifacts.coverage_report)
    LOGGER.info("Class weights: %s", artifacts.class_weights)


if __name__ == "__main__":
    main()
