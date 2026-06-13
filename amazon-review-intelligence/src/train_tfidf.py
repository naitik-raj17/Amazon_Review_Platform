"""Train and evaluate TF-IDF based sentiment classifiers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from data_loader import prepare_dataset, train_test_split_dataframe
from preprocess import TextPreprocessor
from utils import get_model_dir, load_env, setup_logging

logger = setup_logging(__name__)
load_env()

TFIDF_MODEL_PATH = "tfidf_sentiment_model.joblib"
TFIDF_METRICS_PATH = "tfidf_metrics.json"


def build_models() -> Dict[str, Pipeline]:
    """Create TF-IDF pipelines with different classifiers."""
    vectorizer = TfidfVectorizer(
        max_features=int(os.getenv("TFIDF_MAX_FEATURES", "10000")),
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )

    return {
        "logistic_regression": Pipeline(
            [
                ("tfidf", vectorizer),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=int(os.getenv("LR_MAX_ITER", "1000")),
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("tfidf", TfidfVectorizer(
                    max_features=int(os.getenv("TFIDF_MAX_FEATURES", "10000")),
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                )),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=int(os.getenv("RF_N_ESTIMATORS", "200")),
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
) -> Dict[str, Any]:
    """Compute classification metrics."""
    metrics = {
        "model": model_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted")),
        "recall": float(recall_score(y_true, y_pred, average="weighted")),
        "f1_score": float(f1_score(y_true, y_pred, average="weighted")),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, output_dict=True),
    }
    logger.info(
        "%s | accuracy=%.4f | f1=%.4f",
        model_name,
        metrics["accuracy"],
        metrics["f1_score"],
    )
    return metrics


def train_tfidf_models(
    df: Optional[pd.DataFrame] = None,
    test_size: float = 0.2,
) -> Tuple[Pipeline, Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Train TF-IDF models and return the best performing pipeline."""
    if df is None:
        df = prepare_dataset()

    preprocessor = TextPreprocessor()
    df = preprocessor.transform_dataframe(df)

    train_df, test_df = train_test_split_dataframe(df, test_size=test_size)

    x_train = train_df["processed_text"].values
    y_train = train_df["sentiment"].values
    x_test = test_df["processed_text"].values
    y_test = test_df["sentiment"].values

    models = build_models()
    all_metrics: Dict[str, Any] = {}
    best_model: Optional[Pipeline] = None
    best_f1 = -1.0
    best_name = ""

    for name, pipeline in models.items():
        logger.info("Training %s...", name)
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        metrics = evaluate_model(y_test, predictions, name)
        all_metrics[name] = metrics

        if metrics["f1_score"] > best_f1:
            best_f1 = metrics["f1_score"]
            best_model = pipeline
            best_name = name

    assert best_model is not None
    all_metrics["best_model"] = best_name

    model_dir = get_model_dir()
    model_path = model_dir / TFIDF_MODEL_PATH
    metrics_path = model_dir / TFIDF_METRICS_PATH

    joblib.dump(best_model, model_path)
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(all_metrics, file, indent=2)

    logger.info("Saved best model (%s) to %s", best_name, model_path)
    return best_model, all_metrics, train_df, test_df


def main() -> None:
    """CLI entry point for TF-IDF training."""
    try:
        _, metrics, _, _ = train_tfidf_models()
        logger.info("Training complete. Best model: %s", metrics.get("best_model"))
    except Exception as exc:
        logger.exception("TF-IDF training failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
