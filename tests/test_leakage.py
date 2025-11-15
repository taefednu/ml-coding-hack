import numpy as np
import pandas as pd

from src.preprocessing import CVTargetEncoder


def test_cv_target_encoder_bounds():
    df = pd.DataFrame({"cat": ["a", "a", "b", "b", "c", "c"]})
    y = pd.Series([0, 1, 0, 1, 0, 1])
    encoder = CVTargetEncoder(smoothing=5.0, min_samples_leaf=1)
    encoder.fit(df, y)
    transformed = encoder.transform(df)
    assert transformed.shape[0] == df.shape[0]
    assert np.all((transformed >= 0) & (transformed <= 1))
