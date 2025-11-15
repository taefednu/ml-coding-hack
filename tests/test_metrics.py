import numpy as np

from src.metrics import compute_metrics, ks_statistic
from src.drift import compute_psi


def test_metrics_basic():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    report = compute_metrics(y_true, y_prob)
    assert round(report.roc_auc, 4) == 1.0
    assert round(report.gini, 4) == 1.0
    assert round(ks_statistic(y_true, y_prob), 4) == 1.0


def test_psi_detects_shift():
    train = np.array([0.1, 0.2, 0.2, 0.3, 0.5])
    valid = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
    import pandas as pd

    df_train = pd.DataFrame({"score": train})
    df_valid = pd.DataFrame({"score": valid})
    psi_scores = compute_psi(df_train, df_valid)
    assert psi_scores[0].psi > 0.1
