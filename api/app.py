from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from api.schemas import HealthResponse, ScoreRequest, ScoreResponse
from src.calibration import ProbabilityCalibrator
from src.scorecard import ScorecardConfig, pd_to_score
from src.utils import get_logger

LOGGER = get_logger(__name__)
app = FastAPI(title="Credit PD Scoring API", version="0.1.0")

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/best_model.pkl"))
CALIBRATOR_PATH = Path(os.getenv("CALIBRATOR_PATH", "models/calibrator.pkl"))
SCORECARD_CFG = ScorecardConfig()

if not MODEL_PATH.exists():
    LOGGER.warning("Model artifact %s not found. API will raise errors until training is executed.", MODEL_PATH)
    MODEL = None
else:
    MODEL = joblib.load(MODEL_PATH)

CALIBRATOR: Optional[ProbabilityCalibrator]
if CALIBRATOR_PATH.exists():
    CALIBRATOR = ProbabilityCalibrator.load(CALIBRATOR_PATH)
else:
    CALIBRATOR = None


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    status = "ok" if MODEL is not None else "missing-model"
    return HealthResponse(status=status, model=os.path.basename(str(MODEL_PATH)))


@app.post("/score", response_model=ScoreResponse)
def score_endpoint(request: ScoreRequest) -> ScoreResponse:
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model artifact missing")
    df = pd.DataFrame([request.features])
    missing = [col for col in MODEL.feature_cols if col not in df.columns]
    for col in missing:
        df[col] = 0
    df = df[MODEL.feature_cols]
    preds = MODEL.predict_proba(df)
    if CALIBRATOR is not None:
        preds = CALIBRATOR.transform(preds)
    score = float(pd_to_score(preds, SCORECARD_CFG)[0])
    LOGGER.info("Scored request with keys=%s", list(request.features.keys()))
    return ScoreResponse(pd=float(preds[0]), score=score, model=MODEL.name)


@app.get("/docs/examples", response_model=dict)
def example_payload() -> dict:
    return {
        "curl": "curl -X POST http://localhost:8080/score -H 'Content-Type: application/json' -d '{\"features\": {\"age\": 35, \"monthly_income\": 12_000_000, \"dpd_30\": 0}}'",
        "note": "Populate all required features as per training schema",
    }
