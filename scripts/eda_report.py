from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import data_loading  # noqa: E402
from src.utils import ensure_path, get_logger, load_config  # noqa: E402

LOGGER = get_logger(__name__)


def _detect_columns(df: pd.DataFrame, config: Dict[str, any]) -> Dict[str, str]:
    target_col = config["target"].get("column")
    date_col = config["split"].get("date_column")
    if not target_col or "<<" in target_col:
        candidates = data_loading.guess_target_columns(df.columns)
        target_col = candidates[0] if candidates else df.columns[0]
    if not date_col or "<<" in date_col:
        candidates = data_loading.guess_date_columns(df.columns)
        date_col = candidates[0] if candidates else None
    return {"target": target_col, "date": date_col}


def _plot_histograms(df: pd.DataFrame, cols: List[str], reports_dir: Path) -> None:
    if not cols:
        return
    sns.set_theme(style="darkgrid")
    n_rows = len(cols)
    fig, axes = plt.subplots(n_rows, 1, figsize=(8, 3 * n_rows))
    if n_rows == 1:
        axes = [axes]
    for ax, col in zip(axes, cols):
        sns.histplot(df[col].dropna(), bins=30, ax=ax, color="#2a9d8f")
        ax.set_title(f"Distribution of {col}")
    fig.tight_layout()
    path = reports_dir / "numeric_hist.png"
    fig.savefig(path)
    plt.close(fig)
    LOGGER.info("Saved histogram plot to %s", path)


def _plot_time_trend(df: pd.DataFrame, date_col: str, target_col: str, reports_dir: Path) -> None:
    if date_col not in df.columns or target_col not in df.columns:
        return
    tmp = df[[date_col, target_col]].dropna()
    if tmp.empty:
        return
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce", dayfirst=True)
    tmp = tmp.dropna()
    if tmp.empty:
        return
    trend = tmp.groupby(date_col)[target_col].mean().rolling(window=7, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(10, 4))
    trend.plot(ax=ax, color="#e76f51")
    ax.set_title(f"Rolling mean of {target_col} over time")
    ax.set_ylabel(target_col)
    fig.tight_layout()
    path = reports_dir / "target_trend.png"
    fig.savefig(path)
    plt.close(fig)
    LOGGER.info("Saved trend plot to %s", path)


def _frame_to_markdown(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown()
    except Exception:
        return df.to_string()


def render_report(config: dict) -> Path:
    data_dir = Path(config["paths"]["data_dir"])
    reports_dir = ensure_path(config["paths"].get("reports_dir", "artifacts/reports"))
    summaries = data_loading.scan_data_dir(data_dir)
    data_loading.export_inventory_report(data_dir, config["paths"].get("inventory_report", reports_dir / "inventory.json"))
    readable = None
    for summary in summaries:
        try:
            readable = data_loading.read_dataset(summary.path, sample_rows=10000)
            break
        except Exception as exc:
            LOGGER.warning("Failed reading %s: %s", summary.path, exc)
    if readable is None:
        raise RuntimeError("No readable datasets for EDA")
    cols = _detect_columns(readable, config)
    try:
        describe = readable.describe(include="all", datetime_is_numeric=True).transpose()
    except TypeError:
        describe = readable.describe(include="all").transpose()
    report_path = reports_dir / "eda_report.md"
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("# Exploratory Data Analysis\n\n")
        fh.write(f"Источник: `{summary.path.name}`\n\n")
        fh.write("## Сводная статистика\n\n")
        fh.write(_frame_to_markdown(describe))
        fh.write("\n\n## Кандидаты целевой переменной\n")
        fh.write("- " + "\n- ".join(data_loading.guess_target_columns(readable.columns)))
        fh.write("\n\n## Кандидаты временных колонок\n")
        fh.write("- " + "\n- ".join(data_loading.guess_date_columns(readable.columns)))
    numeric_cols = readable.select_dtypes(include=["number"]).columns.tolist()[:4]
    _plot_histograms(readable, numeric_cols, reports_dir)
    if cols.get("date") and cols.get("target") in readable.columns:
        _plot_time_trend(readable, cols["date"], cols["target"], reports_dir)
    LOGGER.info("EDA report saved to %s", report_path)
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate lightweight EDA report")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    render_report(config)


if __name__ == "__main__":
    main()
