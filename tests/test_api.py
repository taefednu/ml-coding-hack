import importlib
import os

import numpy as np

from api.schemas import ScoreRequest


class DummyModel:
    def __init__(self):
        self.name = "dummy"
        self.feature_cols = ["x1", "x2"]

    def predict_proba(self, df):
        return np.array([0.4])


def test_api_health_and_score(tmp_path):
    model_path = tmp_path / "missing.pkl"
    os.environ["MODEL_PATH"] = str(model_path)
    os.environ.pop("CALIBRATOR_PATH", None)
    from api import app as api_app

    importlib.reload(api_app)
    api_app.MODEL = DummyModel()
    api_app.CALIBRATOR = None
    health = api_app.healthz()
    assert health.status == "ok"
    request = ScoreRequest(features={"x1": 10, "x2": 5})
    response = api_app.score_endpoint(request)
    assert 0 <= response.pd <= 1
