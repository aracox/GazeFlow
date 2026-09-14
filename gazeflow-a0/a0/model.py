"""StandardScaler + Ridge regression pipeline for gaze prediction (section 26).

Separate models are fit for x_norm and y_norm; no polynomial features are
used for the A0 baseline.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_pipeline(alpha: float) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha)),
    ])


@dataclass
class GazeModel:
    alpha: float
    model_x: Pipeline
    model_y: Pipeline

    @classmethod
    def fit(cls, X: np.ndarray, y_x: np.ndarray, y_y: np.ndarray, alpha: float) -> "GazeModel":
        mx = build_pipeline(alpha).fit(X, y_x)
        my = build_pipeline(alpha).fit(X, y_y)
        return cls(alpha=alpha, model_x=mx, model_y=my)

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.model_x.predict(X), self.model_y.predict(X)
