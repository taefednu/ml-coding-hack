"""Data inventory utilities for the hackathon pipeline.

The module focuses on *safe* and *deterministic* reading of heterogeneous
files. It automatically tries to infer encodings, delimiters and sensible
schema information so that the downstream steps can rely on a unified view
of the inputs.
"""
from __future__ import annotations

import csv
import io
import json
import re
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import xml.etree.ElementTree as ET

import importlib
import numpy as np
import pandas as pd
from charset_normalizer import from_bytes

try:
    from src.utils import get_logger, guess_columns
except ImportError:  # pragma: no cover - fallback when executed as script
    from utils import get_logger, guess_columns

LOGGER = get_logger(__name__)
SUPPORTED_SUFFIXES = {".csv", ".parquet", ".jsonl", ".json", ".xml", ".xlsx"}
CUSTOMER_ID_CANDIDATES = [
    "customer_ref",
    "customer_id",
    "customer_number",
    "cust_id",
    "cust_num",
    "customer",
]


def _parquet_engine_available() -> bool:
    for module in ("pyarrow", "fastparquet"):
        try:
            importlib.import_module(module)
            return True
        except ImportError:
            continue
    return False


PARQUET_ENGINE_AVAILABLE = _parquet_engine_available()


def _coerce_numeric_series(series: pd.Series) -> pd.Series:
    if series.dtype.kind in {"i", "f"}:
        return pd.to_numeric(series, errors="coerce")
    cleaned = (
        series.astype(str)
        .str.replace(r"[^0-9.+-]", "", regex=True)
        .replace({"": np.nan})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _auto_cast_numeric(df: pd.DataFrame, exclude: Optional[List[str]] = None) -> pd.DataFrame:
    exclude = set(exclude or [])
    for col in df.columns:
        if col in exclude:
            continue
        if df[col].dtype.kind in {"i", "f"}:
            continue
        sample = df[col].dropna().astype(str).head(50)
        if sample.empty:
            continue
        if sample.str.contains(r"[0-9]").mean() >= 0.6:
            df[col] = _coerce_numeric_series(df[col])
    return df


def _standardize_customer_id(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    if id_col in df.columns:
        df[id_col] = df[id_col]
    else:
        for candidate in CUSTOMER_ID_CANDIDATES:
            if candidate in df.columns:
                df[id_col] = df[candidate]
                break
    if id_col not in df.columns:
        raise ValueError(f"Failed to locate customer identifier columns in dataframe columns={df.columns.tolist()}")
    df[id_col] = df[id_col].astype(str)
    drop_cols = [col for col in CUSTOMER_ID_CANDIDATES if col in df.columns and col != id_col]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df


def _ensure_application_date(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    date_col = config.get("split", {}).get("date_column", "application_date")
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        return df
    if "account_open_year" in df.columns:
        df[date_col] = pd.to_datetime(df["account_open_year"].astype(int).astype(str) + "-06-30", errors="coerce")
        return df
    raise ValueError(f"Unable to infer application date column {date_col}")


def _resolve_source_path(config: Dict[str, Any], source_key: str) -> Optional[Path]:
    sources = config.get("data_sources", {})
    filename = sources.get(source_key)
    if not filename:
        return None
    data_dir = Path(config["paths"]["data_dir"])
    path = data_dir / filename
    if not path.exists():
        LOGGER.warning("Source file %s missing at %s", source_key, path)
        return None
    return path


def _load_source(path: Optional[Path]) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    try:
        return read_dataset(path)
    except Exception as exc:
        LOGGER.warning("Failed to read %s: %s", path, exc)
        return pd.DataFrame()


def load_master_dataset(config: Dict[str, Any], use_evaluation: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load master dataset from training or evaluation directory.
    
    Args:
        config: Configuration dictionary
        use_evaluation: If True, load from evaluation_dir instead of data_dir
        
    Returns:
        Tuple of (master_df, credit_history_df)
    """
    # Temporarily override data_dir if using evaluation set
    original_data_dir = config["paths"].get("data_dir")
    if use_evaluation and "evaluation_dir" in config["paths"]:
        config["paths"]["data_dir"] = config["paths"]["evaluation_dir"]
    
    try:
        id_col = config.get("merging", {}).get("id_col", "customer_ref")
        application = _load_source(_resolve_source_path(config, "application"))
        if application.empty:
            raise RuntimeError("Application dataset is required and could not be loaded")
        application = _standardize_customer_id(application, id_col)
        application = _ensure_application_date(application, config)
        demographics = _load_source(_resolve_source_path(config, "demographics"))
        if not demographics.empty:
            demographics = _standardize_customer_id(demographics, id_col)
            demographics = _auto_cast_numeric(demographics, exclude=[id_col])
        loan_details = _load_source(_resolve_source_path(config, "loan_details"))
        if not loan_details.empty:
            loan_details = _standardize_customer_id(loan_details, id_col)
            loan_details = _auto_cast_numeric(loan_details, exclude=[id_col])
        financial = _load_source(_resolve_source_path(config, "financial_ratios"))
        if not financial.empty:
            financial = _standardize_customer_id(financial, id_col)
            financial = _auto_cast_numeric(financial, exclude=[id_col])
        credit_history = _load_source(_resolve_source_path(config, "credit_history"))
        if not credit_history.empty:
            credit_history = _standardize_customer_id(credit_history, id_col)
            credit_history = _auto_cast_numeric(credit_history, exclude=[id_col, config.get("merging", {}).get("credit_history_date_col")])

        master = application.copy()
        if not demographics.empty:
            master = master.merge(demographics, on=id_col, how="left", suffixes=("", "_demo"))
        if not loan_details.empty:
            master = master.merge(loan_details, on=id_col, how="left", suffixes=("", "_loan"))
        if not financial.empty:
            master = master.merge(financial, on=id_col, how="left", suffixes=("", "_fin"))
        return master, credit_history
    finally:
        # Restore original data_dir
        if use_evaluation and original_data_dir:
            config["paths"]["data_dir"] = original_data_dir


def _read_bytes(path: Path, n_bytes: int = 200_000) -> bytes:
    with open(path, "rb") as fh:
        return fh.read(n_bytes)


def detect_encoding(path: str | Path) -> str:
    sample = _read_bytes(Path(path))
    if not sample:
        return "utf-8"
    result = from_bytes(sample).best()
    if result is None:
        return "utf-8"
    return result.encoding or "utf-8"


def detect_delimiter(path: str | Path, encoding: str) -> Optional[str]:
    suffix = Path(path).suffix.lower()
    if suffix not in {".csv", ".txt", ""}:
        return None
    sample_bytes = _read_bytes(Path(path))
    sample_text = sample_bytes.decode(encoding, errors="ignore")
    try:
        dialect = csv.Sniffer().sniff(sample_text[:10_000])
        return dialect.delimiter
    except csv.Error:
        return None


def _read_csv(path: Path, encoding: str, delimiter: Optional[str], nrows: Optional[int]) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.read_csv(
            path,
            encoding=encoding,
            delimiter=delimiter,
            nrows=nrows,
            low_memory=False,
        )


def _read_xlsx_naive(path: Path) -> pd.DataFrame:
    """Minimal XLSX reader that avoids optional dependencies."""

    def strip_ns(tag: str) -> str:
        return tag.split("}")[-1]

    def col_index(cell_ref: str) -> int:
        letters = re.sub(r"[^A-Z]", "", cell_ref.upper())
        col = 0
        for ch in letters:
            col = col * 26 + (ord(ch) - 64)
        return col - 1

    with zipfile.ZipFile(path) as zf:
        sheet_names = sorted(fn for fn in zf.namelist() if fn.startswith("xl/worksheets/sheet"))
        if not sheet_names:
            raise ValueError("No worksheets found in XLSX file")
        sheet_root = ET.fromstring(zf.read(sheet_names[0]))
        shared_strings: List[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            ss_root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in ss_root.iter():
                if strip_ns(si.tag) == "si":
                    text_fragments = [
                        (node.text or "") for node in si.iter() if strip_ns(node.tag) == "t"
                    ]
                    shared_strings.append("".join(text_fragments))

    rows: List[List[Any]] = []
    for row in sheet_root.iter():
        if strip_ns(row.tag) != "row":
            continue
        cells: Dict[int, Any] = {}
        max_col = -1
        for cell in row:
            if strip_ns(cell.tag) != "c":
                continue
            ref = cell.attrib.get("r")
            if ref is None:
                continue
            col_idx = col_index(ref)
            max_col = max(max_col, col_idx)
            cell_type = cell.attrib.get("t")
            value_node = next((child for child in cell if strip_ns(child.tag) == "v"), None)
            value: Any = None
            if cell_type == "inlineStr":
                inline = next((child for child in cell if strip_ns(child.tag) == "is"), None)
                if inline is not None:
                    fragments = [
                        (node.text or "")
                        for node in inline.iter()
                        if strip_ns(node.tag) == "t" and node.text
                    ]
                    value = "".join(fragments) if fragments else None
            elif value_node is not None and value_node.text is not None:
                if cell_type == "s":
                    idx = int(value_node.text)
                    value = shared_strings[idx] if idx < len(shared_strings) else value_node.text
                else:
                    value = value_node.text
            cells[col_idx] = value
        if max_col == -1:
            continue
        row_values = [cells.get(idx) for idx in range(max_col + 1)]
        rows.append(row_values)
    if not rows:
        return pd.DataFrame()
    header = rows[0]
    data = rows[1:] if len(rows) > 1 else []
    df = pd.DataFrame(data, columns=header)
    return df


def _read_tabular(path: Path, encoding: str, delimiter: Optional[str], nrows: Optional[int]) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        if not PARQUET_ENGINE_AVAILABLE:
            raise ImportError("Parquet engine not installed (pyarrow or fastparquet required)")
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return _read_csv(path, encoding, delimiter, nrows)
    if suffix in {".jsonl", ".json"}:
        return pd.read_json(path, encoding=encoding, lines=suffix == ".jsonl", nrows=nrows)
    if suffix == ".xlsx":
        try:
            return pd.read_excel(path, nrows=nrows)
        except ImportError:
            return _read_xlsx_naive(path)
    if suffix == ".xml":
        return pd.read_xml(path, encoding=encoding, parser="etree")
    raise ValueError(f"Unsupported file type: {suffix}")


@dataclass
class ColumnSummary:
    name: str
    dtype: str
    null_fraction: float
    unique: int
    example: Any


@dataclass
class DatasetSummary:
    path: Path
    encoding: str
    delimiter: Optional[str]
    n_rows: Optional[int]
    n_cols: Optional[int]
    schema: List[ColumnSummary]
    preview: pd.DataFrame
    target_candidates: List[str]
    date_candidates: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": str(self.path),
            "encoding": self.encoding,
            "delimiter": self.delimiter,
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "schema": [vars(col) for col in self.schema],
            "preview": self.preview.head().to_dict(orient="records"),
            "target_candidates": self.target_candidates,
            "date_candidates": self.date_candidates,
        }


def _summarize_dataframe(df: pd.DataFrame) -> List[ColumnSummary]:
    summaries: List[ColumnSummary] = []
    for col in df.columns:
        series = df[col]
        null_fraction = float(series.isna().mean()) if len(series) else 0.0
        unique = int(series.nunique(dropna=True)) if len(series) else 0
        example = series.dropna().iloc[0] if series.dropna().empty is False else None
        summaries.append(
            ColumnSummary(
                name=col,
                dtype=str(series.dtype),
                null_fraction=round(null_fraction, 4),
                unique=unique,
                example=example,
            )
        )
    return summaries


def guess_target_columns(columns: Iterable[str]) -> List[str]:
    keywords = ["target", "label", "default", "bad", "overdue", "dpd", "delinq"]
    return list(dict.fromkeys(guess_columns(columns, keywords)))


def guess_date_columns(columns: Iterable[str]) -> List[str]:
    keywords = ["date", "dt", "report", "issue", "created", "time"]
    return list(dict.fromkeys(guess_columns(columns, keywords)))


def build_credit_history_features(
    credit_history: pd.DataFrame,
    master_df: pd.DataFrame,
    config: Dict[str, Any],
) -> pd.DataFrame:
    if credit_history is None or credit_history.empty:
        return pd.DataFrame()
    id_col = config.get("merging", {}).get("id_col", "customer_ref")
    app_col = config.get("split", {}).get("application_id_col", "application_id")
    date_col = config.get("merging", {}).get("credit_history_date_col", "statement_date")
    obs_months = config.get("target", {}).get("observation_window_months")
    reference_date_col = config.get("split", {}).get("date_column", "application_date")
    keys = master_df[[app_col, id_col, reference_date_col]].copy()
    keys = keys.rename(columns={reference_date_col: "__application_date"})
    merged = credit_history.merge(keys, on=id_col, how="right", suffixes=("", "_app"))
    if date_col in merged.columns:
        merged[date_col] = pd.to_datetime(merged[date_col], errors="coerce")
        merged = merged.dropna(subset=["__application_date"])
        if merged[date_col].notna().any():
            merged = merged[merged[date_col] <= merged["__application_date"]]
            if obs_months:
                lower = merged["__application_date"] - pd.DateOffset(months=int(obs_months))
                merged = merged[merged[date_col] >= lower]
            merged = merged.sort_values([app_col, date_col])
    numeric_cols = [
        col
        for col in merged.select_dtypes(include=["number"]).columns
        if col not in {app_col}
    ]
    agg = pd.DataFrame()
    if numeric_cols:
        agg = merged.groupby(app_col, dropna=False)[numeric_cols].agg(["mean", "max", "min"])
        agg.columns = [f"{col}_{stat}_hist" for col, stat in agg.columns]
        agg = agg.reset_index()
    counts = merged.groupby(app_col, dropna=False).size().rename("credit_history_records").reset_index()
    result = keys[[app_col]].drop_duplicates().copy()
    if not agg.empty:
        result = result.merge(agg, on=app_col, how="left")
    result = result.merge(counts, on=app_col, how="left")
    dpd_thresholds = config.get("merging", {}).get("dpd_thresholds", [])
    for threshold in dpd_thresholds:
        col = f"dpd_{threshold}_flag"
        if col in merged.columns:
            stats = merged.groupby(app_col, dropna=False)[col].agg(["sum", "mean", "max"]).reset_index()
            stats = stats.rename(
                columns={
                    "sum": f"{col}_sum_hist",
                    "mean": f"{col}_rate_hist",
                    "max": f"{col}_max_hist",
                }
            )
            result = result.merge(stats, on=app_col, how="left")
    if date_col not in merged.columns or merged[date_col].isna().all():
        return result
    dated = merged.dropna(subset=[date_col]).copy()
    if dated.empty:
        return result
    dpd_cols = [col for col in dated.columns if any(token in col.lower() for token in ("dpd", "delinq", "overdue"))]
    payment_cols = [col for col in dated.columns if "payment" in col.lower()]
    debt_cols = [col for col in dated.columns if "debt" in col.lower() or "balance" in col.lower()]
    util_numerators = [col for col in dated.columns if "credit_usage" in col.lower() or "revolving_balance" in col.lower()]
    util_denominators = [col for col in dated.columns if "limit" in col.lower() or "available_credit" in col.lower()]
    windows = [3, 6, 12]
    for window in windows:
        lower = dated["__application_date"] - pd.DateOffset(months=window)
        mask = dated[date_col] >= lower
        window_df = dated.loc[mask].copy()
        if window_df.empty:
            continue
        group = window_df.groupby(app_col, dropna=False)
        window_features: Dict[str, pd.Series] = {}
        window_features[f"credit_history_records_{window}m"] = group.size()
        if dpd_cols:
            dpd_values = pd.to_numeric(window_df[dpd_cols], errors="coerce").max(axis=1)
            dpd_group = dpd_values.groupby(window_df[app_col])
            window_features[f"dpd_max_{window}m"] = dpd_group.max()
            window_features[f"dpd_mean_{window}m"] = dpd_group.mean()
            window_features[f"num_late_payments_{window}m"] = (dpd_values > 0).groupby(window_df[app_col]).sum()
        if util_numerators and util_denominators:
            num = pd.to_numeric(window_df[util_numerators[0]], errors="coerce")
            den = pd.to_numeric(window_df[util_denominators[0]], errors="coerce").replace(0, np.nan)
            util_ratio = num / den
            util_group = util_ratio.groupby(window_df[app_col])
            window_features[f"utilization_avg_{window}m"] = util_group.mean()
            window_features[f"utilization_max_{window}m"] = util_group.max()
        if payment_cols and debt_cols:
            pay = pd.to_numeric(window_df[payment_cols[0]], errors="coerce")
            debt = pd.to_numeric(window_df[debt_cols[0]], errors="coerce").replace(0, np.nan)
            ratio = pay / debt
            window_features[f"payment_to_debt_ratio_{window}m"] = ratio.groupby(window_df[app_col]).mean()
        if window == 6 and debt_cols:
            balance = pd.to_numeric(window_df[debt_cols[0]], errors="coerce")
            ordered = window_df.assign(_balance=balance).dropna(subset=["_balance"])
            if not ordered.empty:
                def _trend(group_df: pd.DataFrame) -> float:
                    series = group_df.sort_values(date_col)["_balance"]
                    if len(series) < 2:
                        return 0.0
                    return float(series.iloc[-1] - series.iloc[0])
                trend = ordered.groupby(app_col, dropna=False).apply(_trend)
                window_features["trend_balance_6m"] = trend
        if window == 6 and payment_cols:
            pay_series = pd.to_numeric(window_df[payment_cols[0]], errors="coerce")
            volatility = pay_series.groupby(window_df[app_col]).std(ddof=0)
            window_features["volatility_payments_6m"] = volatility
        if window_features:
            window_df_final = pd.DataFrame(window_features)
            window_df_final.index.name = app_col
            window_df_final = window_df_final.reset_index()
            result = result.merge(window_df_final, on=app_col, how="left")
    
    # Advanced behavioral features if enabled
    feat_cfg = config.get("feature_engineering", {})
    if feat_cfg.get("advanced_behavioral_features", False) and not dated.empty:
        advanced_features = _build_advanced_behavioral_features(
            dated, app_col, date_col, dpd_cols, payment_cols, debt_cols, 
            util_numerators, util_denominators
        )
        if not advanced_features.empty:
            result = result.merge(advanced_features, on=app_col, how="left")
    
    return result


def _build_advanced_behavioral_features(
    dated: pd.DataFrame,
    app_col: str,
    date_col: str,
    dpd_cols: List[str],
    payment_cols: List[str],
    debt_cols: List[str],
    util_numerators: List[str],
    util_denominators: List[str],
) -> pd.DataFrame:
    """Build advanced behavioral features: DPD counters, trends, volatility, stress indicators."""
    unique_apps = dated[app_col].unique()
    result = pd.DataFrame({app_col: unique_apps})
    
    windows = [3, 6, 12]
    
    for window in windows:
        lower = dated["__application_date"] - pd.DateOffset(months=window)
        mask = dated[date_col] >= lower
        window_df = dated.loc[mask].copy().sort_values([app_col, date_col])
        
        if window_df.empty:
            continue
        
        group = window_df.groupby(app_col, dropna=False)
        advanced: Dict[str, pd.Series] = {}
        
        # DPD counters
        if dpd_cols:
            dpd_values = pd.to_numeric(window_df[dpd_cols], errors="coerce").max(axis=1).fillna(0)
            
            # Count months with DPD > 0, >= 30, >= 60
            advanced[f"beh_dpd_cnt_gt0_{window}m"] = (dpd_values > 0).groupby(window_df[app_col]).sum()
            advanced[f"beh_dpd_cnt_ge30_{window}m"] = (dpd_values >= 30).groupby(window_df[app_col]).sum()
            advanced[f"beh_dpd_cnt_ge60_{window}m"] = (dpd_values >= 60).groupby(window_df[app_col]).sum()
            
            # Maximum DPD
            advanced[f"beh_dpd_max_{window}m"] = dpd_values.groupby(window_df[app_col]).max()
            
            # Stress indicators: proportion of months with DPD > 0
            month_counts = group.size()
            dpd_months = (dpd_values > 0).groupby(window_df[app_col]).sum()
            advanced[f"beh_dpd_rate_{window}m"] = (dpd_months / month_counts).fillna(0.0)
            
            # Recovery episodes: transitions from DPD>0 to DPD=0
            if window >= 12:
                def _count_recoveries(group_df: pd.DataFrame) -> int:
                    dpd_series = pd.to_numeric(group_df[dpd_cols].max(axis=1), errors="coerce").fillna(0)
                    was_dpd = (dpd_series > 0).astype(int)
                    # Shift to detect transitions: 1 -> 0 means recovery
                    shifted = was_dpd.shift(1).fillna(0)
                    recoveries = ((was_dpd == 0) & (shifted == 1)).sum()
                    return int(recoveries)
                
                advanced[f"beh_recovery_episodes_{window}m"] = group.apply(_count_recoveries)
        
        # Trends: linear slope for DTI/debt_service_ratio and utilization
        if debt_cols and payment_cols and window >= 6:
            balance = pd.to_numeric(window_df[debt_cols[0]], errors="coerce")
            payment = pd.to_numeric(window_df[payment_cols[0]], errors="coerce").replace(0, np.nan)
            
            if balance.notna().any() and payment.notna().any():
                def _calculate_slope(group_df: pd.DataFrame, col: str) -> float:
                    series = group_df.sort_values(date_col)[col].dropna()
                    if len(series) < 2:
                        return 0.0
                    x = np.arange(len(series))
                    slope = np.polyfit(x, series.values, 1)[0] if len(series) > 1 else 0.0
                    return float(slope)
                
                balance_slope = window_df.assign(_balance=balance).dropna(subset=["_balance"]).groupby(app_col, dropna=False).apply(
                    lambda g: _calculate_slope(g, "_balance"), include_groups=False
                )
                advanced[f"beh_balance_slope_{window}m"] = balance_slope
                
                if window >= 12:
                    payment_slope = window_df.assign(_payment=payment).dropna(subset=["_payment"]).groupby(app_col, dropna=False).apply(
                        lambda g: _calculate_slope(g, "_payment"), include_groups=False
                    )
                    advanced[f"beh_payment_slope_{window}m"] = payment_slope
        
        # Utilization trends
        if util_numerators and util_denominators and window >= 6:
            num = pd.to_numeric(window_df[util_numerators[0]], errors="coerce")
            den = pd.to_numeric(window_df[util_denominators[0]], errors="coerce").replace(0, np.nan)
            util_ratio = (num / den).fillna(0.0)
            
            def _calculate_slope_util(group_df: pd.DataFrame) -> float:
                series = group_df.sort_values(date_col)["_util"].dropna()
                if len(series) < 2:
                    return 0.0
                x = np.arange(len(series))
                slope = np.polyfit(x, series.values, 1)[0] if len(series) > 1 else 0.0
                return float(slope)
            
            util_slope = window_df.assign(_util=util_ratio).dropna(subset=["_util"]).groupby(app_col, dropna=False).apply(
                _calculate_slope_util, include_groups=False
            )
            advanced[f"beh_util_slope_{window}m"] = util_slope
        
        # Deltas: last_value - mean_value
        if debt_cols:
            balance = pd.to_numeric(window_df[debt_cols[0]], errors="coerce")
            balance_mean = balance.groupby(window_df[app_col]).mean()
            balance_last = window_df.assign(_balance=balance).dropna(subset=["_balance"]).groupby(app_col, dropna=False).apply(
                lambda g: g.sort_values(date_col)["_balance"].iloc[-1], include_groups=False
            )
            advanced[f"beh_balance_delta_{window}m"] = balance_last - balance_mean
        
        if payment_cols:
            payment = pd.to_numeric(window_df[payment_cols[0]], errors="coerce")
            payment_mean = payment.groupby(window_df[app_col]).mean()
            payment_last = window_df.assign(_payment=payment).dropna(subset=["_payment"]).groupby(app_col, dropna=False).apply(
                lambda g: g.sort_values(date_col)["_payment"].iloc[-1], include_groups=False
            )
            advanced[f"beh_payment_delta_{window}m"] = payment_last - payment_mean
        
        # Volatility: std and coefficient of variation
        if payment_cols:
            payment = pd.to_numeric(window_df[payment_cols[0]], errors="coerce")
            payment_std = payment.groupby(window_df[app_col]).std(ddof=0).fillna(0.0)
            payment_mean = payment.groupby(window_df[app_col]).mean().replace(0, np.nan)
            advanced[f"beh_payment_std_{window}m"] = payment_std
            advanced[f"beh_payment_cv_{window}m"] = (payment_std / payment_mean).fillna(0.0)
        
        if debt_cols:
            balance = pd.to_numeric(window_df[debt_cols[0]], errors="coerce")
            balance_std = balance.groupby(window_df[app_col]).std(ddof=0).fillna(0.0)
            balance_mean = balance.groupby(window_df[app_col]).mean().replace(0, np.nan)
            advanced[f"beh_balance_std_{window}m"] = balance_std
            advanced[f"beh_balance_cv_{window}m"] = (balance_std / balance_mean).fillna(0.0)
        
        if util_numerators and util_denominators:
            num = pd.to_numeric(window_df[util_numerators[0]], errors="coerce")
            den = pd.to_numeric(window_df[util_denominators[0]], errors="coerce").replace(0, np.nan)
            util_ratio = (num / den).fillna(0.0)
            util_std = util_ratio.groupby(window_df[app_col]).std(ddof=0).fillna(0.0)
            util_mean = util_ratio.groupby(window_df[app_col]).mean().replace(0, np.nan)
            advanced[f"beh_util_std_{window}m"] = util_std
            advanced[f"beh_util_cv_{window}m"] = (util_std / util_mean).fillna(0.0)
        
        # Stress: proportion of months with payment < minimum (if available)
        if payment_cols:
            payment = pd.to_numeric(window_df[payment_cols[0]], errors="coerce")
            # Look for minimum payment column
            min_payment_cols = [col for col in window_df.columns if "min" in col.lower() and "payment" in col.lower()]
            if min_payment_cols:
                min_payment = pd.to_numeric(window_df[min_payment_cols[0]], errors="coerce")
                underpaid = (payment < min_payment).fillna(False)
                month_counts = group.size()
                underpaid_months = underpaid.groupby(window_df[app_col]).sum()
                advanced[f"beh_underpayment_rate_{window}m"] = (underpaid_months / month_counts).fillna(0.0)
        
        # Merge advanced features
        if advanced:
            advanced_df = pd.DataFrame(advanced)
            advanced_df.index.name = app_col
            advanced_df = advanced_df.reset_index()
            result = result.merge(advanced_df, on=app_col, how="left")
    
    return result


def read_dataset(path: str | Path, sample_rows: Optional[int] = None) -> pd.DataFrame:
    path = Path(path)
    encoding = detect_encoding(path)
    delimiter = detect_delimiter(path, encoding)
    df = _read_tabular(path, encoding, delimiter, sample_rows)
    return df


def inventory_dataset(path: str | Path, sample_rows: int = 5000) -> DatasetSummary:
    path = Path(path)
    encoding = detect_encoding(path)
    delimiter = detect_delimiter(path, encoding)
    df = _read_tabular(path, encoding, delimiter, sample_rows)
    schema = _summarize_dataframe(df)
    target_candidates = guess_target_columns(df.columns)
    date_candidates = guess_date_columns(df.columns)
    n_rows = None
    n_cols = None
    try:
        n_rows = int(df.shape[0])
        n_cols = int(df.shape[1])
    except Exception:
        pass
    LOGGER.info("Inventoried %s (encoding=%s, delimiter=%s)", path.name, encoding, delimiter)
    return DatasetSummary(
        path=path,
        encoding=encoding,
        delimiter=delimiter,
        n_rows=n_rows,
        n_cols=n_cols,
        schema=schema,
        preview=df.head(5),
        target_candidates=target_candidates,
        date_candidates=date_candidates,
    )


def scan_data_dir(data_dir: str | Path) -> List[DatasetSummary]:
    data_dir = Path(data_dir)
    summaries: List[DatasetSummary] = []
    for path in data_dir.rglob("*"):
        if path.is_dir():
            continue
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            LOGGER.debug("Skipping unsupported file: %s", path)
            continue
        if path.suffix.lower() == ".parquet" and not PARQUET_ENGINE_AVAILABLE:
            LOGGER.info("Skipping %s (Parquet engine not installed)", path.name)
            continue
        try:
            summaries.append(inventory_dataset(path))
        except Exception as exc:
            LOGGER.warning("Failed to inventory %s: %s", path, exc)
    return summaries


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        return obj.isoformat()
    return str(obj)


def export_inventory_report(data_dir: str | Path, output: str | Path) -> Path:
    summaries = scan_data_dir(data_dir)
    payload = [summary.to_dict() for summary in summaries]
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, default=_json_default)
    LOGGER.info("Inventory report saved to %s", path)
    return path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inventory datasets in data directory")
    parser.add_argument("data_dir", type=str, help="Directory with raw files")
    parser.add_argument("--output", type=str, default="artifacts/data_inventory.json")
    args = parser.parse_args()
    export_inventory_report(args.data_dir, args.output)


__all__ = [
    "ColumnSummary",
    "DatasetSummary",
    "detect_delimiter",
    "detect_encoding",
    "build_credit_history_features",
    "export_inventory_report",
    "guess_date_columns",
    "guess_target_columns",
    "load_master_dataset",
    "inventory_dataset",
    "read_dataset",
    "scan_data_dir",
]
