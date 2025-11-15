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
from typing import Any, Dict, Iterable, List, Optional
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from charset_normalizer import from_bytes

try:
    from src.utils import get_logger, guess_columns
except ImportError:  # pragma: no cover - fallback when executed as script
    from utils import get_logger, guess_columns

LOGGER = get_logger(__name__)
SUPPORTED_SUFFIXES = {".csv", ".parquet", ".jsonl", ".json", ".xml", ".xlsx"}


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
    "export_inventory_report",
    "guess_date_columns",
    "guess_target_columns",
    "inventory_dataset",
    "read_dataset",
    "scan_data_dir",
]
