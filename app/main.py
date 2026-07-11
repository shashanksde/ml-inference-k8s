"""FastAPI application serving handwritten-digit predictions.

Endpoints:
    GET  /healthz   liveness  — 200 whenever the process is up
    GET  /readyz    readiness — 200 only once the model is loaded
    GET  /metadata  model info (type, feature count, labels, training accuracy)
    POST /predict   run inference on a flattened 8x8 image (64 pixels, 0-16)

The health endpoints are wired to the Kubernetes liveness/readiness probes in
``k8s/deployment.yaml``.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app import __version__
from app.model import N_FEATURES, load_model, predict

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("ml-inference-k8s")

# Populated at startup by the lifespan handler.
_state: dict = {"model": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load (or train) the model once when the service starts."""
    logger.info("Loading model ...")
    _state["model"] = load_model()
    logger.info("Model ready (test accuracy=%.4f)", _state["model"].accuracy)
    yield
    _state["model"] = None


app = FastAPI(
    title="ml-inference-k8s",
    description="Handwritten-digit (0-9) inference service.",
    version=__version__,
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    pixels: List[float] = Field(
        ...,
        min_length=N_FEATURES,
        max_length=N_FEATURES,
        description=f"Flattened 8x8 image: exactly {N_FEATURES} pixel values (0-16).",
    )


class PredictResponse(BaseModel):
    prediction: int
    confidence: float
    probabilities: List[float]


@app.get("/healthz")
def healthz() -> dict:
    """Liveness probe: the process is running."""
    return {"status": "ok"}


@app.get("/readyz")
def readyz():
    """Readiness probe: the model has finished loading."""
    if _state["model"] is None:
        return JSONResponse(status_code=503, content={"status": "loading"})
    return {"status": "ready"}


@app.get("/metadata")
def metadata():
    model = _state["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {
        "model_type": model.model_type,
        "n_features": model.n_features,
        "labels": model.labels,
        "training_accuracy": model.accuracy,
        "version": __version__,
    }


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(request: PredictRequest) -> PredictResponse:
    model = _state["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    result = predict(model, request.pixels)
    return PredictResponse(**result)
