"""Data merger and credit history aggregation for the FE pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.feature_eng import build_features
from src.utils import get_logger

LOGGER = get_logger(__name__)


def _match_column(columns: Sequence[str], keywords: Sequence[str]) -> Optional[str]:
    lower = {col.lower(): col for col in columns}
    for keyword in keywords:
        if keyword in lower:
            return lower[keyword]
    for col in columns:
        lower_name = col.lower()
        if any(keyword in lower_name for keyword in keywords):
            return col
    return None


def _safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    num = pd.to_numeric(num, errors="coerce")
    den = pd.to_numeric(den, errors="coerce")
    den = den.mask(den == 0)
    ratio = num / den
    return ratio.replace([np.inf, -np.inf], np.nan)


@dataclass
class MergeConfig:
    id_col: str = "customer_ref"
    application_id_col: str = "application_id"
    application_datetime_col: Optional[str] = None
    credit_history_date_col: Optional[str] = None
    history_spec: Optional[dict] = None
    dpd_thresholds: Tuple[int, int, int] = (30, 60, 90)


def prepare_application_metadata(df: pd.DataFrame, config: MergeConfig) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("Application metadata cannot be empty")
    df = df.copy()
    if config.id_col not in df.columns:
        raise ValueError(f"Application data must contain '{config.id_col}'")
    df[config.id_col] = df[config.id_col].astype(str).str.strip()
    if "default" not in df.columns:
        candidate = _match_column(df.columns, ["default", "target", "bad_flag"])
        if not candidate:
            raise ValueError("Could not locate default/target column in application metadata")
        df["default"] = df[candidate]
        if candidate != "default":
            df = df.drop(columns=[candidate])
    df["default"] = pd.to_numeric(df["default"], errors="coerce").fillna(0).astype(int)
    if config.application_datetime_col and config.application_datetime_col in df.columns:
        df[config.application_datetime_col] = pd.to_datetime(
            df[config.application_datetime_col], errors="coerce"
        )
    return df


def _deduplicate(df: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    present_keys = [key for key in keys if key in df.columns]
    if not present_keys:
        return df
    return df.sort_values(present_keys).drop_duplicates(subset=present_keys, keep="last")


def build_credit_history_aggregates(
    history_df: Optional[pd.DataFrame],
    config: MergeConfig,
) -> pd.DataFrame:
    if history_df is None or history_df.empty:
        return pd.DataFrame(columns=[config.id_col])
    if config.id_col not in history_df.columns:
        raise ValueError(f"Credit history is missing '{config.id_col}' column")
    df = history_df.copy()
    entity_col = config.id_col
    date_col = config.credit_history_date_col or _match_column(
        df.columns, ["date", "statement", "report", "period"]
    )
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    else:
        date_col = "__synthetic_date__"
        df[date_col] = pd.Timestamp.utcnow()

    dpd_col = _match_column(df.columns, ["dpd", "days_past_due"])
    if dpd_col:
        dpd = pd.to_numeric(df[dpd_col], errors="coerce").fillna(0)
        df["dpd_value"] = dpd
        for thr in config.dpd_thresholds:
            df[f"dpd_{thr}_flag"] = (dpd >= thr).astype(int)
    balance_col = _match_column(df.columns, ["balance", "outstanding"])
    due_col = _match_column(df.columns, ["due", "installment", "due_amount"])
    payment_col = _match_column(df.columns, ["payment", "paid", "received"])
    limit_col = _match_column(df.columns, ["limit", "credit_limit"])

    if payment_col and due_col:
        df["payment_ratio"] = _safe_ratio(df[payment_col], df[due_col])
    if balance_col and limit_col:
        df["utilization"] = _safe_ratio(df[balance_col], df[limit_col])
    if balance_col and due_col:
        df["past_due_ratio"] = _safe_ratio(df[balance_col] - df.get(payment_col, 0), df[due_col])

    group = df.groupby(entity_col)
    summary = pd.DataFrame(index=group.size().index)

    agg_map: Dict[str, List[str]] = {}
    for base_col in ["dpd_value", "payment_ratio", "utilization", "past_due_ratio"]:
        if base_col in df.columns:
            agg_map[base_col] = ["mean", "max", "std"]
    if agg_map:
        agg_df = group.agg(agg_map)
        agg_df.columns = [f"{col}_{stat}" for col, stat in agg_df.columns]
        summary = summary.join(agg_df, how="left")

    for thr in config.dpd_thresholds:
        flag_col = f"dpd_{thr}_flag"
        if flag_col not in df.columns:
            continue
        summary[f"{flag_col}_count"] = group[flag_col].sum()
        summary[f"{flag_col}_share"] = group[flag_col].mean()
        summary[f"{flag_col}_chronic"] = (group[flag_col].mean() > 0.5).astype(int)

    history_features = build_features(
        df,
        spec_config=config.history_spec,
        date_col=date_col,
        entity_col=entity_col,
    )
    if not history_features.empty:
        ordered = df.sort_values([entity_col, date_col])
        last_idx = ordered.groupby(entity_col).tail(1).index
        last_features = history_features.loc[last_idx]
        last_features[entity_col] = ordered.loc[last_idx, entity_col].values
        last_features = last_features.set_index(entity_col)
        summary = summary.join(last_features, how="left")

    summary.reset_index(inplace=True)
    summary.rename(columns={"index": entity_col}, inplace=True)
    return summary


def _merge(
    base: pd.DataFrame,
    df: Optional[pd.DataFrame],
    keys: List[str],
    name: str,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    if df is None or df.empty:
        return base, {"coverage": 0.0, "rows": 0}
    df = _deduplicate(df, keys)
    present_keys = [key for key in keys if key in df.columns and key in base.columns]
    if not present_keys:
        LOGGER.warning("Skipping merge for %s – no common keys", name)
        return base, {"coverage": 0.0, "rows": len(df)}
    merged = base.merge(df, how="left", on=present_keys, suffixes=("", f"_{name}"))
    coverage = (merged[present_keys[0]].isin(df[present_keys[0]])).mean()
    stats = {"coverage": float(coverage), "rows": float(len(df))}
    return merged, stats


def build_master_table(
    application_df: pd.DataFrame,
    demographics_df: Optional[pd.DataFrame] = None,
    loan_df: Optional[pd.DataFrame] = None,
    ratios_df: Optional[pd.DataFrame] = None,
    history_df: Optional[pd.DataFrame] = None,
    config: Optional[MergeConfig] = None,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    cfg = config or MergeConfig()
    base = prepare_application_metadata(application_df, cfg)
    coverage_report: Dict[str, Dict[str, float]] = {}

    merge_plan = [
        ("demographics", demographics_df, [cfg.id_col]),
        ("loan_details", loan_df, [cfg.id_col, cfg.application_id_col]),
        ("financial_ratios", ratios_df, [cfg.id_col]),
    ]
    for name, frame, keys in merge_plan:
        base, stats = _merge(base, frame, keys, name)
        coverage_report[name] = stats

    history_agg = build_credit_history_aggregates(history_df, cfg)
    base, stats = _merge(base, history_agg, [cfg.id_col], "credit_history")
    coverage_report["credit_history"] = stats

    return base, coverage_report


__all__ = [
    "MergeConfig",
    "build_credit_history_aggregates",
    "build_master_table",
    "prepare_application_metadata",
]
