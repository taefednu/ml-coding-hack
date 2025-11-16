"""End-to-end preprocessing + feature engineering orchestrator."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pandas as pd

from src.id_normalization import IDNormalizer
from src.merging import MergeConfig, build_master_table
from src.preprocessing import AdvancedPreprocessor, PreprocessingConfig
from src.data_loading import build_credit_history_features
from src.utils import get_logger

LOGGER = get_logger(__name__)


@dataclass
class PipelineArtifacts:
    master_table: pd.DataFrame
    features: pd.DataFrame
    target: pd.Series
    preprocessor: AdvancedPreprocessor
    coverage_report: Dict[str, Dict[str, float]]
    class_weights: Dict[int, float]


def preprocess_and_generate_features(
    application_df: pd.DataFrame,
    demographics_df: Optional[pd.DataFrame] = None,
    loan_df: Optional[pd.DataFrame] = None,
    ratios_df: Optional[pd.DataFrame] = None,
    history_df: Optional[pd.DataFrame] = None,
    merge_config: Optional[MergeConfig] = None,
    preprocessing_config: Optional[PreprocessingConfig] = None,
    feature_engineering_config: Optional[Dict] = None,
    return_artifacts: bool = False,
) -> Tuple[pd.DataFrame, Optional[PipelineArtifacts]] | pd.DataFrame:
    """Main entry point for notebooks and scripts."""

    normalizer = IDNormalizer()
    normalized_frames = {
        "application": normalizer.normalize(application_df, "application"),
        "demographics": normalizer.normalize(demographics_df, "demographics") if demographics_df is not None else None,
        "loan_details": normalizer.normalize(loan_df, "loan_details") if loan_df is not None else None,
        "financial_ratios": normalizer.normalize(ratios_df, "financial_ratios") if ratios_df is not None else None,
        "credit_history": normalizer.normalize(history_df, "credit_history") if history_df is not None else None,
    }

    merge_cfg = merge_config or MergeConfig()
    master_table, coverage_report = build_master_table(
        normalized_frames["application"],
        demographics_df=normalized_frames["demographics"],
        loan_df=normalized_frames["loan_details"],
        ratios_df=normalized_frames["financial_ratios"],
        history_df=normalized_frames["credit_history"],
        config=merge_cfg,
    )
    
    # Add advanced credit history features if enabled
    feat_cfg = feature_engineering_config or {}
    if feat_cfg.get("advanced_behavioral_features", False) and not normalized_frames["credit_history"].empty:
        # Build config dict for build_credit_history_features
        config_dict = {
            "merging": {
                "id_col": merge_cfg.id_col,
                "credit_history_date_col": merge_cfg.credit_history_date_col or "statement_date",
                "dpd_thresholds": list(merge_cfg.dpd_thresholds),
            },
            "split": {
                "date_column": merge_cfg.application_datetime_col or "application_date",
                "application_id_col": merge_cfg.application_id_col,
            },
            "target": {
                "observation_window_months": 12,
            },
            "feature_engineering": feat_cfg,
        }
        
        # Add application date column if not present
        date_col = config_dict["split"]["date_column"]
        if date_col not in master_table.columns and merge_cfg.application_datetime_col in master_table.columns:
            master_table[date_col] = master_table[merge_cfg.application_datetime_col]
        
        # Build advanced credit history features
        credit_features = build_credit_history_features(
            normalized_frames["credit_history"],
            master_table,
            config_dict,
        )
        if not credit_features.empty:
            app_col = merge_cfg.application_id_col
            if app_col in master_table.columns and app_col in credit_features.columns:
                master_table = master_table.merge(credit_features, on=app_col, how="left")
                LOGGER.info("Added %d advanced credit history features", len(credit_features.columns) - 1)
    prep_cfg = preprocessing_config or PreprocessingConfig()
    target_col = prep_cfg.target_col
    if target_col not in master_table.columns:
        raise ValueError(f"Target column '{target_col}' missing from master table")
    target = master_table[target_col].astype(float)

    preprocessor = AdvancedPreprocessor(prep_cfg)
    features = preprocessor.fit_transform(master_table, y=target)
    dataset = features.copy()
    dataset[target_col] = target.values

    artifacts = PipelineArtifacts(
        master_table=master_table,
        features=features,
        target=target,
        preprocessor=preprocessor,
        coverage_report=coverage_report,
        class_weights=preprocessor.class_weights_,
    )

    if return_artifacts:
        return dataset, artifacts
    dataset.attrs["artifacts"] = artifacts
    return dataset


__all__ = ["PipelineArtifacts", "preprocess_and_generate_features"]
