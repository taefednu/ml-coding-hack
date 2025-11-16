from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel, Field


class ScoreRequest(BaseModel):
    features: Dict[str, Any] = Field(..., description="Raw feature dictionary matching training schema")


class ScoreResponse(BaseModel):
    pd: float
    score: float
    model: str


class HealthResponse(BaseModel):
    status: str
    model: str
