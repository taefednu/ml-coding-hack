"""Utility helpers for configuration, logging, and reproducibility.

The helpers here are intentionally lightweight so that other modules can
import them without triggering heavy dependencies. Only drop extensions
into this file if they are broadly useful across the project.
"""
from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import yaml

DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


@dataclass
class ArtifactPaths:
    """Bundle paths to persistently stored artefacts."""

    models_dir: Path
    artifacts_dir: Path
    reports_dir: Path

    def ensure(self) -> "ArtifactPaths":
        for path in (self.models_dir, self.artifacts_dir, self.reports_dir):
            path.mkdir(parents=True, exist_ok=True)
        return self


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(config_path: str | Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    return config


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def get_logger(name: str, level: str | int = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        # torch is optional
        pass


def ensure_path(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def resolve_path(path: str | Path, base_dir: Optional[str | Path] = None) -> Path:
    p = Path(path)
    if not p.is_absolute() and base_dir is not None:
        p = Path(base_dir) / p
    return p


def read_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def column_exists(columns: Iterable[str], candidate: str | None) -> bool:
    return candidate is not None and candidate in set(columns)


def guess_columns(columns: Iterable[str], keywords: Iterable[str]) -> list[str]:
    columns_set = [col.lower() for col in columns]
    matches: list[str] = []
    for keyword in keywords:
        for original, lower in zip(columns, columns_set):
            if keyword in lower:
                matches.append(original)
    return matches


__all__ = [
    "ArtifactPaths",
    "DEFAULT_LOG_FORMAT",
    "column_exists",
    "ensure_path",
    "get_logger",
    "guess_columns",
    "load_config",
    "project_root",
    "read_yaml",
    "resolve_path",
    "save_json",
    "seed_everything",
]
