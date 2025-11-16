"""Scorecard helpers: 300-900 scale conversion and export."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml

from src.utils import get_logger

LOGGER = get_logger(__name__)


@dataclass
class ScorecardConfig:
    pdo: float = 50.0
    base_odds: float = 20.0
    base_score: float = 650.0
    min_score: int = 300
    max_score: int = 900

    @property
    def factor(self) -> float:
        return self.pdo / np.log(2)

    @property
    def offset(self) -> float:
        return self.base_score - self.factor * np.log(self.base_odds)


def pd_to_score(pd_values: np.ndarray, cfg: ScorecardConfig) -> np.ndarray:
    pd_safe = np.clip(pd_values, 1e-6, 1 - 1e-6)
    odds = (1 - pd_safe) / pd_safe
    score = cfg.offset + cfg.factor * np.log(odds)
    return np.clip(score, cfg.min_score, cfg.max_score)


def score_to_pd(scores: np.ndarray, cfg: ScorecardConfig) -> np.ndarray:
    odds = np.exp((scores - cfg.offset) / cfg.factor)
    return 1 / (1 + odds)


def _to_native(obj):
    if isinstance(obj, dict):
        return {str(key): _to_native(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_to_native(item) for item in obj]
    if isinstance(obj, tuple):
        return [_to_native(item) for item in obj]
    if isinstance(obj, np.ndarray):
        return [_to_native(item) for item in obj.tolist()]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def export_scorecard(
    feature_points: Dict[str, float],
    binning_tables: Optional[List[Dict[str, float]]],
    path: str | Path,
    cfg: ScorecardConfig | None = None,
) -> Path:
    cfg = cfg or ScorecardConfig()
    payload = _to_native({
        "points_to_odds": {
            "factor": cfg.factor,
            "offset": cfg.offset,
        },
        "config": asdict(cfg),
        "feature_points": feature_points,
        "woe_bins": binning_tables or [],
    })
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".yaml", ".yml"}:
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(payload, fh, allow_unicode=True, sort_keys=False)
    else:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
    LOGGER.info("Scorecard exported to %s", path)
    return path


__all__ = ["ScorecardConfig", "pd_to_score", "score_to_pd", "export_scorecard"]
