"""FastAPI backend for sentiment prediction."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

from predict import SentimentPredictor  # noqa: E402
from utils import load_env, setup_logging  # noqa: E402

load_env()
logger = setup_logging("api")

app = FastAPI(
    title="Amazon Customer Review Intelligence Platform",
    description="Real-time sentiment analysis for Amazon customer reviews",
    version="1.0.0",
)

_predictor: Optional[SentimentPredictor] = None


class PredictRequest(BaseModel):
    review: str = Field(
        ...,
        min_length=3,
        max_length=5000,
        description="Review text to classify",
        examples=["Product arrived late and packaging was damaged"],
    )


class PredictResponse(BaseModel):
    sentiment: str
    confidence: float


class HealthResponse(BaseModel):
    status: str
    message: str
    model_type: str


def get_predictor() -> SentimentPredictor:
    """Lazy-load the sentiment predictor."""
    global _predictor
    if _predictor is None:
        model_type = os.getenv("DEFAULT_MODEL_TYPE", "transformer")
        _predictor = SentimentPredictor(model_type=model_type)  # type: ignore[arg-type]
        logger.info("Predictor initialized with model_type=%s", model_type)
    return _predictor


@app.get("/", response_model=HealthResponse)
def root() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        message="Amazon Customer Review Intelligence Platform API is running",
        model_type=os.getenv("DEFAULT_MODEL_TYPE", "transformer"),
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Predict sentiment for a given review."""
    try:
        predictor = get_predictor()
        result = predictor.predict(request.review)
        return PredictResponse(**result)
    except ValueError as exc:
        logger.warning("Validation error: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        logger.error("Model not found: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Model not available. Train models before making predictions.",
        ) from exc
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail="Internal prediction error.") from exc


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run("app:app", host=host, port=port, reload=True)
