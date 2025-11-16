"""Identifier harmonisation utilities.

The scoring inputs originate from different vendor systems and therefore
expose inconsistent identifier names (`client_id`, `cust_num`, etc.). The
helpers in this module coerce everything into `customer_ref`, which the rest
of the pipeline expects.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import pandas as pd

from src.utils import get_logger, guess_columns

LOGGER = get_logger(__name__)

DEFAULT_ID_ALIASES: Dict[str, List[str]] = {
    "application": ["customer_ref", "client_id", "customer_id", "cust_id", "customer_ref_id"],
    "demographics": ["customer_ref", "cust_id", "customer_id", "client_ref"],
    "loan_details": ["customer_ref", "customer_id", "cust_ref", "client_id"],
    "financial_ratios": ["cust_num", "customer_ref", "customer_id", "client_id"],
    "credit_history": ["customer_ref", "customer_id", "cust_id", "client_ref", "customer_number"],
}


def _pick_first_match(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    lower = {col.lower(): col for col in columns}
    for name in candidates:
        if name in lower:
            return lower[name]
    for name in candidates:
        matches = guess_columns(columns, [name])
        if matches:
            return matches[0]
    return None


@dataclass
class IDNormalizer:
    """Normalise different identifier names to a single `customer_ref`."""

    target_column: str = "customer_ref"
    alias_mapping: Dict[str, List[str]] = field(default_factory=lambda: DEFAULT_ID_ALIASES)
    cast_to_str: bool = True
    strip_chars: bool = True

    def normalize(self, df: pd.DataFrame, dataset_name: str, id_column: Optional[str] = None) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        df = df.copy()
        aliases = self.alias_mapping.get(dataset_name.lower(), [])
        candidate_column = id_column or _pick_first_match(df.columns, aliases)
        if candidate_column is None:
            raise ValueError(f"Could not find customer identifier in dataset '{dataset_name}'")
        LOGGER.info("Normalising %s identifier '%s' -> '%s'", dataset_name, candidate_column, self.target_column)
        series = df[candidate_column]
        if self.cast_to_str:
            series = series.astype(str)
        if self.strip_chars:
            series = series.str.strip()
        df[self.target_column] = series
        if candidate_column != self.target_column:
            df = df.drop(columns=[candidate_column])
        return df

    def normalize_many(self, frames: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        normalised: Dict[str, pd.DataFrame] = {}
        for name, frame in frames.items():
            if frame is None:
                normalised[name] = frame
                continue
            normalised[name] = self.normalize(frame, dataset_name=name)
        return normalised


__all__ = ["IDNormalizer", "DEFAULT_ID_ALIASES"]
