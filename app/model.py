"""Model training, persistence, and loading.

We use scikit-learn's bundled ``load_digits`` dataset (8x8 grayscale handwritten
digits, pixel values 0-16) so the whole project trains in seconds and needs no
network access. It is the same task as MNIST — classify a digit 0-9 — just at a
smaller resolution, which keeps the demo reproducible and offline-friendly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

import joblib
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Number of input features: an 8x8 image flattened to 64 pixels.
N_FEATURES = 64
LABELS: List[int] = list(range(10))

DEFAULT_MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "..", "digits_model.joblib"),
)


@dataclass
class TrainedModel:
    """A fitted estimator bundled with the metadata we expose over the API."""

    estimator: Pipeline
    accuracy: float
    model_type: str
    n_features: int
    labels: List[int]


def train_model(random_state: int = 42) -> TrainedModel:
    """Train a StandardScaler + LogisticRegression pipeline on the digits dataset."""
    digits = load_digits()
    X, y = digits.data, digits.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    estimator = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(max_iter=2000, random_state=random_state),
            ),
        ]
    )
    estimator.fit(X_train, y_train)
    accuracy = float(estimator.score(X_test, y_test))

    return TrainedModel(
        estimator=estimator,
        accuracy=accuracy,
        model_type="StandardScaler + LogisticRegression",
        n_features=N_FEATURES,
        labels=LABELS,
    )


def save_model(model: TrainedModel, path: str = DEFAULT_MODEL_PATH) -> str:
    """Persist a :class:`TrainedModel` to disk with joblib. Returns the path written."""
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path: str = DEFAULT_MODEL_PATH) -> TrainedModel:
    """Load a persisted model, or train + save one on the fly if none exists.

    This keeps the service self-sufficient: even if the artifact is missing (e.g.
    a fresh checkout), the app can still come up and serve predictions.
    """
    path = os.path.abspath(path)
    if os.path.exists(path):
        return joblib.load(path)
    model = train_model()
    save_model(model, path)
    return model


def predict(model: TrainedModel, pixels: List[float]) -> dict:
    """Run inference on a single flattened 8x8 image (64 pixel values, 0-16)."""
    if len(pixels) != N_FEATURES:
        raise ValueError(f"expected {N_FEATURES} pixel values, got {len(pixels)}")
    x = np.asarray(pixels, dtype=float).reshape(1, -1)
    proba = model.estimator.predict_proba(x)[0]
    prediction = int(np.argmax(proba))
    return {
        "prediction": prediction,
        "confidence": float(proba[prediction]),
        "probabilities": [float(p) for p in proba],
    }
