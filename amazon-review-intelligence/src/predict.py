"""Unified prediction interface for sentiment models."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

import joblib
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from preprocess import TextPreprocessor
from utils import get_model_dir, load_env, setup_logging

logger = setup_logging(__name__)
load_env()

ModelType = Literal["tfidf", "transformer"]

TFIDF_MODEL_PATH = "tfidf_sentiment_model.joblib"
TRANSFORMER_MODEL_DIR = "distilbert_sentiment"


class SentimentPredictor:
    """Load and run sentiment predictions using TF-IDF or transformer models."""

    def __init__(
        self,
        model_type: Optional[ModelType] = None,
        model_dir: Optional[Path] = None,
    ) -> None:
        self.model_type: ModelType = (
            model_type or os.getenv("DEFAULT_MODEL_TYPE", "transformer")  # type: ignore[assignment]
        )
        self.model_dir = model_dir or get_model_dir()
        self.preprocessor = TextPreprocessor()
        self._tfidf_model = None
        self._transformer_model = None
        self._tokenizer = None
        self._label_map: Dict[int, str] = {0: "Negative", 1: "Positive"}

    def _load_tfidf(self) -> None:
        if self._tfidf_model is not None:
            return
        model_path = self.model_dir / TFIDF_MODEL_PATH
        if not model_path.exists():
            raise FileNotFoundError(
                f"TF-IDF model not found at {model_path}. Run train_tfidf.py first."
            )
        self._tfidf_model = joblib.load(model_path)
        logger.info("Loaded TF-IDF model from %s", model_path)

    def _load_transformer(self) -> None:
        if self._transformer_model is not None:
            return
        model_path = self.model_dir / TRANSFORMER_MODEL_DIR
        if not model_path.exists():
            raise FileNotFoundError(
                f"Transformer model not found at {model_path}. "
                "Run train_transformer.py first."
            )
        self._tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self._transformer_model = AutoModelForSequenceClassification.from_pretrained(
            str(model_path)
        )
        self._transformer_model.eval()

        label_map_path = model_path / "label_map.json"
        if label_map_path.exists():
            with open(label_map_path, encoding="utf-8") as file:
                raw_map = json.load(file)
            self._label_map = {int(v): k for k, v in raw_map.items()}

        logger.info("Loaded transformer model from %s", model_path)

    def predict_tfidf(self, review: str) -> Tuple[str, float]:
        """Predict sentiment using the TF-IDF pipeline."""
        self._load_tfidf()
        processed = self.preprocessor.clean_text(review)
        prediction = self._tfidf_model.predict([processed])[0]

        if hasattr(self._tfidf_model, "predict_proba"):
            proba = self._tfidf_model.predict_proba([processed])[0]
            confidence = float(np.max(proba))
        else:
            confidence = 1.0

        return str(prediction), confidence

    def predict_transformer(self, review: str) -> Tuple[str, float]:
        """Predict sentiment using the fine-tuned DistilBERT model."""
        self._load_transformer()
        assert self._tokenizer is not None
        assert self._transformer_model is not None

        inputs = self._tokenizer(
            review,
            truncation=True,
            padding=True,
            max_length=int(os.getenv("MAX_SEQ_LENGTH", "128")),
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = self._transformer_model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1).numpy()[0]

        predicted_id = int(np.argmax(probabilities))
        sentiment = self._label_map.get(predicted_id, "Negative")
        confidence = float(probabilities[predicted_id])
        return sentiment, confidence

    def predict(self, review: str) -> Dict[str, Any]:
        """Predict sentiment using the configured model type."""
        if not review or not review.strip():
            raise ValueError("Review text cannot be empty.")

        try:
            if self.model_type == "tfidf":
                sentiment, confidence = self.predict_tfidf(review)
            else:
                sentiment, confidence = self.predict_transformer(review)
        except FileNotFoundError:
            logger.warning(
                "%s model unavailable, falling back to TF-IDF", self.model_type
            )
            sentiment, confidence = self.predict_tfidf(review)

        return {
            "sentiment": sentiment,
            "confidence": round(confidence, 4),
        }


def predict_sentiment(
    review: str,
    model_type: Optional[ModelType] = None,
) -> Dict[str, Any]:
    """Convenience function for single-review prediction."""
    predictor = SentimentPredictor(model_type=model_type)
    return predictor.predict(review)
