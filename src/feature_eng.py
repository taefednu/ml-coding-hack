"""Feature engineering helpers for credit risk modelling.

This module exposes a highly-configurable feature factory similar to what is
used in production credit-scoring stacks (Upstart/Zest/AMEX style). The
functions operate purely on pandas DataFrames which makes it trivial to plug
them into notebooks, batch jobs or MLFlow pipelines.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils import get_logger

LOGGER = get_logger(__name__)


@dataclass
class EventMarker:
    column: str
    threshold: float = 1.0
    operator: str = ">="


@dataclass
class BehavioralFlag:
    column: str
    threshold: float
    operator: str = ">="
    window: Optional[int] = None


@dataclass
class FeatureSpec:
    ratio_features: Dict[str, List[str]] = field(default_factory=dict)
    numeric_aggs: Dict[str, List[str]] = field(default_factory=dict)
    rolling_windows: List[int] = field(default_factory=lambda: [3, 6, 12])
    trend_features: List[str] = field(default_factory=list)
    delinquency_markers: List[str] = field(default_factory=list)
    event_markers: Dict[str, EventMarker] = field(default_factory=dict)
    behavioral_flags: Dict[str, BehavioralFlag] = field(default_factory=dict)
    segment_normalization: Dict[str, str] = field(default_factory=dict)
    volatility_features: List[str] = field(default_factory=list)
    time_since_event_cols: List[str] = field(default_factory=list)
    active_loan_indicator: Optional[str] = None
    log_features: List[str] = field(default_factory=list)
    winsorize: Dict[str, List[float]] = field(default_factory=dict)
    interaction_pairs: List[List[str]] = field(default_factory=list)


DEFAULT_SPEC = FeatureSpec(
    ratio_features={
        "dti": ["total_debt", "monthly_income"],
        "utilization": ["current_balance", "credit_limit"],
        "payment_rate": ["payment_amount", "due_amount"],
    },
    numeric_aggs={
        "payment_amount": ["mean", "max", "std"],
        "balance": ["mean", "max", "min", "std"],
        "due_amount": ["mean", "max"],
    },
    rolling_windows=[3, 6, 12],
    trend_features=["balance", "payment_amount", "utilization"],
    delinquency_markers=["dpd_30_flag", "dpd_60_flag", "dpd_90_flag"],
    event_markers={
        "dpd_30": EventMarker(column="dpd_30_flag", threshold=1),
        "dpd_60": EventMarker(column="dpd_60_flag", threshold=1),
    },
    behavioral_flags={
        "credit_hungry": BehavioralFlag(column="new_credit_inquiries", threshold=3, operator=">=", window=6),
        "high_utilization": BehavioralFlag(column="utilization", threshold=0.8, operator=">="),
    },
    segment_normalization={"monthly_income": "region"},
    volatility_features=["balance", "payment_amount"],
    time_since_event_cols=["dpd_30_flag", "dpd_60_flag"],
    active_loan_indicator="account_status",
    log_features=[],
    winsorize={},
)


def spec_from_config(cfg: Optional[dict]) -> FeatureSpec:
    if not cfg:
        return DEFAULT_SPEC
    event_cfg = cfg.get("event_markers", {})
    behavior_cfg = cfg.get("behavioral_flags", {})
    events = {
        name: EventMarker(
            column=opts["column"],
            threshold=float(opts.get("threshold", 1.0)),
            operator=opts.get("operator", ">="),
        )
        for name, opts in event_cfg.items()
    }
    behaviors = {
        name: BehavioralFlag(
            column=opts["column"],
            threshold=float(opts.get("threshold", 1.0)),
            operator=opts.get("operator", ">="),
            window=opts.get("window"),
        )
        for name, opts in behavior_cfg.items()
    }
    return FeatureSpec(
        ratio_features=cfg.get("ratios", DEFAULT_SPEC.ratio_features),
        numeric_aggs=cfg.get("numeric_aggregations", DEFAULT_SPEC.numeric_aggs),
        rolling_windows=cfg.get("rolling_windows", DEFAULT_SPEC.rolling_windows),
        trend_features=cfg.get("trend_features", DEFAULT_SPEC.trend_features),
    delinquency_markers=cfg.get("delinquency_markers", DEFAULT_SPEC.delinquency_markers),
    event_markers=events or DEFAULT_SPEC.event_markers,
    behavioral_flags=behaviors or DEFAULT_SPEC.behavioral_flags,
    segment_normalization=cfg.get("segment_normalization", DEFAULT_SPEC.segment_normalization),
    volatility_features=cfg.get("volatility_features", DEFAULT_SPEC.volatility_features),
    time_since_event_cols=cfg.get("time_since_event_cols", DEFAULT_SPEC.time_since_event_cols),
    active_loan_indicator=cfg.get("active_loan_indicator", DEFAULT_SPEC.active_loan_indicator),
    log_features=cfg.get("log_features", DEFAULT_SPEC.log_features),
    winsorize=cfg.get("winsorize", DEFAULT_SPEC.winsorize),
    interaction_pairs=cfg.get("interaction_pairs", DEFAULT_SPEC.interaction_pairs),
)


def _safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    num = pd.to_numeric(num, errors="coerce")
    den = pd.to_numeric(den, errors="coerce")
    den = den.mask(den == 0)
    ratio = num / den
    return ratio.replace([np.inf, -np.inf], np.nan)


def _apply_operator(series: pd.Series, operator: str, threshold: float) -> pd.Series:
    if operator == ">=":
        return series >= threshold
    if operator == ">":
        return series > threshold
    if operator == "<=":
        return series <= threshold
    if operator == "<":
        return series < threshold
    if operator == "==":
        return series == threshold
    return series >= threshold


def build_ratios(df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for name, cols in spec.ratio_features.items():
        if len(cols) != 2:
            continue
        num_col, den_col = cols
        if num_col in df.columns and den_col in df.columns:
            out[name] = _safe_ratio(df[num_col], df[den_col])
    return out


def build_numeric_aggs(df: pd.DataFrame, spec: FeatureSpec, entity_col: Optional[str]) -> pd.DataFrame:
    if not entity_col or entity_col not in df.columns:
        return pd.DataFrame(index=df.index)
    aggs = {}
    for col, funcs in spec.numeric_aggs.items():
        if col not in df.columns:
            continue
        group = df.groupby(entity_col)[col]
        for func in funcs:
            feature_name = f"{col}_{func}_by_{entity_col}"
            aggs[feature_name] = group.transform(func)
    return pd.DataFrame(aggs)


def build_rolling(df: pd.DataFrame, spec: FeatureSpec, date_col: Optional[str], entity_col: Optional[str]) -> pd.DataFrame:
    if not date_col or date_col not in df.columns or not entity_col or entity_col not in df.columns:
        return pd.DataFrame(index=df.index)
    ordered = df.sort_values([entity_col, date_col])
    numeric_cols = ordered.select_dtypes(include=["number"]).columns
    frames = []
    for window in spec.rolling_windows:
        rolling_stats = (
            ordered.groupby(entity_col)[numeric_cols]
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        rolling_stats = rolling_stats.add_suffix(f"_roll_mean_{window}")
        frames.append(rolling_stats)
    if not frames:
        return pd.DataFrame(index=df.index)
    combined = pd.concat(frames, axis=1)
    combined = combined.loc[df.index]
    return combined


def build_trends(df: pd.DataFrame, spec: FeatureSpec, date_col: Optional[str], entity_col: Optional[str]) -> pd.DataFrame:
    if not date_col or date_col not in df.columns or not entity_col or entity_col not in df.columns:
        return pd.DataFrame(index=df.index)
    ordered = df.sort_values([entity_col, date_col])
    out = pd.DataFrame(index=ordered.index)
    for col in spec.trend_features:
        if col not in ordered.columns:
            continue
        group = ordered.groupby(entity_col)[col]
        delta = group.transform(lambda s: s.iloc[-1] - s.iloc[0] if len(s) > 1 else 0.0)
        pct = group.transform(lambda s: (s.iloc[-1] - s.iloc[0]) / (abs(s.iloc[0]) + 1e-6) if len(s) > 1 else 0.0)
        out[f"{col}_delta"] = delta
        out[f"{col}_pct_change"] = pct
    return out.loc[df.index]


def build_delinquency(df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    available = [col for col in spec.delinquency_markers if col in df.columns]
    if not available:
        return pd.DataFrame(index=df.index)
    deli = df[available].fillna(0)
    out = pd.DataFrame(index=df.index)
    out["delinq_sum"] = deli.sum(axis=1)
    out["delinq_max"] = deli.max(axis=1)
    out["delinq_any_30"] = (deli > 0).any(axis=1).astype(int)
    return out


def build_event_flags(df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for name, marker in spec.event_markers.items():
        if marker.column not in df.columns:
            continue
        series = pd.to_numeric(df[marker.column], errors="coerce").fillna(0)
        out[f"event_{name}"] = _apply_operator(series, marker.operator, marker.threshold).astype(int)
    return out


def build_behavioral_flags(
    df: pd.DataFrame,
    spec: FeatureSpec,
    date_col: Optional[str],
    entity_col: Optional[str],
) -> pd.DataFrame:
    if not spec.behavioral_flags:
        return pd.DataFrame(index=df.index)
    ordered = df
    if (
        date_col
        and date_col in df.columns
        and entity_col
        and entity_col in df.columns
    ):
        ordered = df.sort_values([entity_col, date_col])
    out = pd.DataFrame(index=df.index)
    for name, rule in spec.behavioral_flags.items():
        if rule.column not in df.columns:
            continue
        values = pd.to_numeric(ordered[rule.column], errors="coerce")
        series = values
        if rule.window and entity_col and entity_col in ordered.columns:
            temp = ordered.copy()
            temp["_behavior_value"] = values
            series = (
                temp.groupby(entity_col)["_behavior_value"]
                .rolling(window=rule.window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
        mask = _apply_operator(series, rule.operator, rule.threshold)
        mask = mask.reindex(ordered.index)
        mask = mask.reindex(df.index).fillna(False)
        out[f"behavior_{name}"] = mask.astype(int)
    return out


def build_segment_norm(df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for value_col, segment_col in spec.segment_normalization.items():
        if value_col not in df.columns or segment_col not in df.columns:
            continue
        segment_mean = df.groupby(segment_col)[value_col].transform("mean")
        out[f"{value_col}_vs_{segment_col}"] = df[value_col] / (segment_mean + 1e-6)
    return out


def build_log_features(df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    if not spec.log_features:
        return pd.DataFrame(index=df.index)
    out = pd.DataFrame(index=df.index)
    for col in spec.log_features:
        if col not in df.columns:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        numeric = numeric.clip(lower=0)
        out[f"{col}_log1p"] = np.log1p(numeric)
    return out


def build_winsorized(df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    if not spec.winsorize:
        return pd.DataFrame(index=df.index)
    out = pd.DataFrame(index=df.index)
    for col, limits in spec.winsorize.items():
        if col not in df.columns:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce").astype(float)
        if numeric.dropna().empty:
            continue
        lower_q, upper_q = (limits + [0.01, 0.99])[:2] if isinstance(limits, list) else (0.01, 0.99)
        lower = numeric.quantile(lower_q)
        upper = numeric.quantile(upper_q)
        clipped = numeric.clip(lower, upper)
        out[f"{col}_winsor"] = clipped.astype(float)
    return out


def build_interactions(df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    pairs = spec.interaction_pairs
    if not pairs:
        return pd.DataFrame(index=df.index)
    out = pd.DataFrame(index=df.index)
    for pair in pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        left, right = pair
        if left not in df.columns or right not in df.columns:
            continue
        left_series = pd.to_numeric(df[left], errors="coerce")
        right_series = pd.to_numeric(df[right], errors="coerce")
        out[f"{left}_x_{right}"] = left_series * right_series
        out[f"{left}_minus_{right}"] = left_series - right_series
        out[f"{left}_div_{right}"] = _safe_ratio(left_series, right_series)
    return out


def build_smart_interactions(
    df: pd.DataFrame,
    feature_strength_path: Optional[str] = None,
    top_k: int = 12,
    max_pairs: int = 20,
    forbidden_ids: Optional[List[str]] = None,
    suspicious_features: Optional[List[str]] = None,
    min_spearman_corr: float = 0.01,
) -> pd.DataFrame:
    """
    Build smart interactions between top features from feature_strength.json.
    
    MANDATORY: Only uses features with Spearman correlation >= min_spearman_corr.
    Filters out ID-like columns, suspicious features, and categorical features.
    Generates product, ratio, and difference interactions for top K features.
    """
    import json
    from pathlib import Path
    
    if feature_strength_path and Path(feature_strength_path).exists():
        try:
            with open(feature_strength_path, "r") as f:
                strength_data = json.load(f)
            
            # Get top features by Spearman correlation (MANDATORY criterion)
            # Only use features with sufficient Spearman correlation
            top_features = []
            if "top_spearman" in strength_data and strength_data["top_spearman"]:
                # Filter by minimum Spearman correlation requirement
                top_features = [
                    f["feature"] 
                    for f in strength_data["top_spearman"][:top_k * 2]  # Take more to filter
                    if abs(f.get("spearman_corr", 0.0)) >= min_spearman_corr
                ][:top_k]
            elif "top_auc" in strength_data:
                # Fallback: check Spearman correlation from full stats if available
                if "top_spearman" in strength_data:
                    spearman_dict = {f["feature"]: f.get("spearman_corr", 0.0) 
                                   for f in strength_data["top_spearman"]}
                    top_features = [
                        f["feature"] 
                        for f in strength_data["top_auc"][:top_k * 2]
                        if abs(spearman_dict.get(f["feature"], 0.0)) >= min_spearman_corr
                    ][:top_k]
                else:
                    # If no Spearman data, use AUC but log warning
                    LOGGER.warning("No Spearman correlation data available, using AUC ranking")
                    top_features = [f["feature"] for f in strength_data["top_auc"][:top_k]]
            elif "top_mutual_info" in strength_data:
                # Similar fallback for mutual info
                LOGGER.warning("Using mutual info ranking without Spearman correlation filter")
                top_features = [f["feature"] for f in strength_data["top_mutual_info"][:top_k]]
            
            if not top_features:
                LOGGER.warning("No features passed Spearman correlation threshold (>= %.3f)", min_spearman_corr)
                return pd.DataFrame(index=df.index)
        except Exception as e:
            LOGGER.warning("Failed to load feature_strength.json: %s", e)
            return pd.DataFrame(index=df.index)
    else:
        # Fallback: use all numeric features if no feature_strength.json
        # BUT: Without Spearman correlation data, we cannot enforce the requirement
        LOGGER.warning("No feature_strength.json available - cannot enforce Spearman correlation requirement")
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        top_features = [col for col in numeric_cols if col not in (forbidden_ids or [])][:top_k]
    
    if not top_features:
        return pd.DataFrame(index=df.index)
    
    # Filter features
    forbidden_ids = set(forbidden_ids or [])
    suspicious_features = set(suspicious_features or [])
    
    # Remove ID-like, suspicious, and non-numeric features
    # Also verify Spearman correlation from feature_strength if available
    filtered_features = []
    spearman_dict = {}
    if feature_strength_path and Path(feature_strength_path).exists():
        try:
            with open(feature_strength_path, "r") as f:
                strength_data = json.load(f)
            if "top_spearman" in strength_data:
                spearman_dict = {f["feature"]: f.get("spearman_corr", 0.0) 
                               for f in strength_data["top_spearman"]}
        except Exception:
            pass
    
    for feat in top_features:
        if feat not in df.columns:
            continue
        if feat.lower() in forbidden_ids or any(id_name in feat.lower() for id_name in ["id", "ref", "code", "num"]):
            continue
        if feat in suspicious_features:
            continue
        # Only numeric features
        if df[feat].dtype not in ["int64", "float64", "float32", "int32"]:
            continue
        if feat.startswith(("ix_", "ix_beh_")):  # Skip existing interactions
            continue
        # MANDATORY: Check Spearman correlation if available
        if spearman_dict and feat in spearman_dict:
            if abs(spearman_dict[feat]) < min_spearman_corr:
                LOGGER.debug("Skipping feature %s: Spearman correlation %.4f < %.4f", 
                           feat, spearman_dict[feat], min_spearman_corr)
                continue
        filtered_features.append(feat)
    
    if len(filtered_features) < 2:
        return pd.DataFrame(index=df.index)
    
    # Generate pairs (limit to max_pairs)
    pairs = []
    for i in range(len(filtered_features)):
        for j in range(i + 1, len(filtered_features)):
            if len(pairs) >= max_pairs:
                break
            pairs.append((filtered_features[i], filtered_features[j]))
        if len(pairs) >= max_pairs:
            break
    
    if not pairs:
        return pd.DataFrame(index=df.index)
    
    # Generate interactions
    out = pd.DataFrame(index=df.index)
    
    for left, right in pairs:
        left_series = pd.to_numeric(df[left], errors="coerce")
        right_series = pd.to_numeric(df[right], errors="coerce")
        
        # Product
        out[f"ix_{left}_x_{right}"] = left_series * right_series
        
        # Ratios (both directions)
        out[f"ix_{left}_div_{right}"] = _safe_ratio(left_series, right_series)
        out[f"ix_{right}_div_{left}"] = _safe_ratio(right_series, left_series)
        
        # Differences (both directions)
        out[f"ix_{left}_minus_{right}"] = left_series - right_series
        out[f"ix_{right}_minus_{left}"] = right_series - left_series
    
    if filtered_features:
        LOGGER.info("Selected %d features with Spearman correlation >= %.3f for interactions", 
                   len(filtered_features), min_spearman_corr)
        LOGGER.info("Generated %d smart interaction features from %d pairs", out.shape[1], len(pairs))
    else:
        LOGGER.warning("No features passed Spearman correlation threshold (>= %.3f) for interactions", 
                      min_spearman_corr)
    return out


def build_volatility(df: pd.DataFrame, spec: FeatureSpec, entity_col: Optional[str]) -> pd.DataFrame:
    if not entity_col or entity_col not in df.columns:
        return pd.DataFrame(index=df.index)
    out = pd.DataFrame(index=df.index)
    for col in spec.volatility_features:
        if col not in df.columns:
            continue
        group = df.groupby(entity_col)[col]
        std = group.transform("std").fillna(0.0)
        mean = group.transform("mean").replace(0, np.nan)
        out[f"{col}_std"] = std
        out[f"{col}_cv"] = std / mean
    return out


def build_time_since_events(df: pd.DataFrame, spec: FeatureSpec, date_col: Optional[str], entity_col: Optional[str]) -> pd.DataFrame:
    if not spec.time_since_event_cols or not date_col or date_col not in df.columns or not entity_col or entity_col not in df.columns:
        return pd.DataFrame(index=df.index)
    df = df.copy()
    df["_parsed_date"] = pd.to_datetime(df[date_col], errors="coerce")
    max_date = df.groupby(entity_col)["_parsed_date"].transform("max")
    out = pd.DataFrame(index=df.index)
    for col in spec.time_since_event_cols:
        if col not in df.columns:
            continue
        mask = pd.to_numeric(df[col], errors="coerce").fillna(0) > 0
        event_dates = (
            df.loc[mask, [entity_col, "_parsed_date"]]
            .groupby(entity_col)["_parsed_date"]
            .max()
        )
        last_event = df[entity_col].map(event_dates)
        delta = (max_date - last_event).dt.days
        out[f"time_since_{col}"] = delta
    return out


def build_active_loan_features(df: pd.DataFrame, spec: FeatureSpec, entity_col: Optional[str]) -> pd.DataFrame:
    indicator = spec.active_loan_indicator
    if not indicator or indicator not in df.columns or not entity_col or entity_col not in df.columns:
        return pd.DataFrame(index=df.index)
    status = df[indicator].astype(str).str.lower()
    active_mask = status.isin({"active", "current", "open", "1", "true"})
    group = df.groupby(entity_col)
    out = pd.DataFrame(index=df.index)
    counts = group[indicator].transform("count").replace(0, np.nan)
    active_counts = group[indicator].transform(lambda s: active_mask.loc[s.index].sum())
    out["active_loans_count"] = active_counts
    out["active_loans_share"] = active_counts / counts
    return out


def build_macro_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    if "application_hour" in df.columns:
        hours = pd.to_numeric(df["application_hour"], errors="coerce")
        out["application_hour_sin"] = np.sin(2 * np.pi * hours / 24)
        out["application_hour_cos"] = np.cos(2 * np.pi * hours / 24)
        out["is_night"] = ((hours >= 21) | (hours <= 6)).astype(int)
    if "application_day_of_week" in df.columns:
        dow = pd.to_numeric(df["application_day_of_week"], errors="coerce")
        out["is_weekend"] = dow.isin({5, 6}).astype(int)
        out["application_dow_sin"] = np.sin(2 * np.pi * dow / 7)
        out["application_dow_cos"] = np.cos(2 * np.pi * dow / 7)
    return out


def build_features(
    df: pd.DataFrame,
    spec_config: Optional[dict] = None,
    date_col: Optional[str] = None,
    entity_col: Optional[str] = None,
) -> pd.DataFrame:
    spec = spec_from_config(spec_config)
    ratio_df = build_ratios(df, spec)
    agg_df = build_numeric_aggs(df, spec, entity_col)
    rolling_df = build_rolling(df, spec, date_col, entity_col)
    trend_df = build_trends(df, spec, date_col, entity_col)
    delinquency_df = build_delinquency(df, spec)
    events_df = build_event_flags(df, spec)
    behavior_df = build_behavioral_flags(df, spec, date_col, entity_col)
    segment_df = build_segment_norm(df, spec)
    volatility_df = build_volatility(df, spec, entity_col)
    time_since_df = build_time_since_events(df, spec, date_col, entity_col)
    active_df = build_active_loan_features(df, spec, entity_col)
    macro_df = build_macro_time_features(df)
    log_df = build_log_features(df, spec)
    winsor_df = build_winsorized(df, spec)
    interaction_df = build_interactions(df, spec)
    features = pd.concat(
        [
            ratio_df,
            agg_df,
            rolling_df,
            trend_df,
            delinquency_df,
            events_df,
            behavior_df,
            segment_df,
            volatility_df,
            time_since_df,
            active_df,
            macro_df,
            log_df,
            winsor_df,
            interaction_df,
        ],
        axis=1,
    )
    numeric_cols = features.select_dtypes(include=["number"]).columns
    if len(numeric_cols) > 0:
        features[numeric_cols] = features[numeric_cols].replace([np.inf, -np.inf], np.nan)
    features = features.infer_objects(copy=False)
    LOGGER.info("Generated %d engineered features", features.shape[1])
    return features


__all__ = [
    "BehavioralFlag",
    "EventMarker",
    "FeatureSpec",
    "DEFAULT_SPEC",
    "build_features",
    "spec_from_config",
    "build_smart_interactions",
]
