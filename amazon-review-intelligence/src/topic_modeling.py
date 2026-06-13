"""BERTopic-based topic modeling for customer reviews."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

from data_loader import prepare_dataset
from preprocess import TextPreprocessor, build_combined_text_column
from utils import get_model_dir, load_env, setup_logging

logger = setup_logging(__name__)
load_env()

TOPIC_MODEL_PATH = "bertopic_model"
TOPIC_REPORT_PATH = "topic_report.json"

EXPECTED_THEMES = [
    "Delivery",
    "Packaging",
    "Product Quality",
    "Pricing",
    "Refunds",
    "Customer Service",
]

THEME_KEYWORDS = {
    "Delivery": ["delivery", "shipping", "ship", "late", "arrived", "courier", "dispatch"],
    "Packaging": ["packaging", "package", "box", "damaged", "wrap", "packed"],
    "Product Quality": ["quality", "product", "material", "build", "durable", "broken"],
    "Pricing": ["price", "pricing", "cost", "expensive", "cheap", "value", "money"],
    "Refunds": ["refund", "return", "replacement", "exchange", "money back"],
    "Customer Service": ["service", "support", "customer", "response", "help", "agent"],
}


def map_topic_to_theme(topic_words: List[str]) -> str:
    """Map a topic's top words to the closest predefined theme."""
    scores = {}
    word_set = set(word.lower() for word in topic_words)
    for theme, keywords in THEME_KEYWORDS.items():
        scores[theme] = sum(1 for kw in keywords if kw in word_set)
    best_theme = max(scores, key=scores.get)
    return best_theme if scores[best_theme] > 0 else "General"


def generate_topic_report(topic_model: BERTopic) -> Dict[str, Any]:
    """Generate a structured topic report with theme mapping."""
    topic_info = topic_model.get_topic_info()
    report: Dict[str, Any] = {"topics": [], "theme_summary": {}}

    for _, row in topic_info.iterrows():
        topic_id = int(row["Topic"])
        if topic_id == -1:
            continue

        words = topic_model.get_topic(topic_id)
        if not words:
            continue

        top_words = [word for word, _ in words[:10]]
        theme = map_topic_to_theme(top_words)
        entry = {
            "topic_id": topic_id,
            "count": int(row["Count"]),
            "top_words": top_words,
            "mapped_theme": theme,
        }
        report["topics"].append(entry)
        report["theme_summary"].setdefault(theme, []).append(entry)

    for theme in EXPECTED_THEMES:
        report["theme_summary"].setdefault(theme, [])

    return report


def train_topic_model(
    df: Optional[pd.DataFrame] = None,
    min_topic_size: Optional[int] = None,
) -> tuple[BERTopic, Dict[str, Any]]:
    """Train BERTopic on review texts and generate topic report."""
    if df is None:
        df = prepare_dataset()

    df = build_combined_text_column(df)
    preprocessor = TextPreprocessor()
    documents = preprocessor.transform(df["combined_text"].astype(str).tolist())

    vectorizer = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=max(1, min(5, len(documents) // 20)),
    )
    topic_size = min_topic_size or int(os.getenv("MIN_TOPIC_SIZE", "10"))
    topic_size = min(topic_size, max(5, len(documents) // 10))

    topic_model = BERTopic(
        min_topic_size=topic_size,
        vectorizer_model=vectorizer,
        verbose=True,
    )

    logger.info("Training BERTopic on %s documents...", len(documents))
    topics, _ = topic_model.fit_transform(documents)
    df = df.copy()
    df["topic"] = topics

    report = generate_topic_report(topic_model)

    model_dir = get_model_dir()
    model_path = model_dir / TOPIC_MODEL_PATH

    topic_model.save(str(model_path))

    report_path = model_dir / TOPIC_REPORT_PATH
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    logger.info("Topic model saved to %s", model_path)
    logger.info("Topic report saved to %s", report_path)
    return topic_model, report


def load_topic_model(model_path: Optional[Path] = None) -> BERTopic:
    """Load a saved BERTopic model."""
    path = model_path or get_model_dir() / TOPIC_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(f"Topic model not found at {path}")
    return BERTopic.load(str(path))


def main() -> None:
    """CLI entry point for topic modeling."""
    try:
        _, report = train_topic_model()
        logger.info("Discovered %s topics", len(report["topics"]))
        for theme, entries in report["theme_summary"].items():
            if entries:
                logger.info("  %s: %s topic(s)", theme, len(entries))
    except Exception as exc:
        logger.exception("Topic modeling failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
