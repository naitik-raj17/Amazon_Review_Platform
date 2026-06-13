"""Fine-tune DistilBERT for sentiment classification."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from data_loader import prepare_dataset, train_test_split_dataframe
from preprocess import build_combined_text_column
from utils import get_model_dir, load_env, setup_logging

logger = setup_logging(__name__)
load_env()

TRANSFORMER_MODEL_NAME = os.getenv("TRANSFORMER_MODEL_NAME", "distilbert-base-uncased")
TRANSFORMER_OUTPUT_DIR = "distilbert_sentiment"
TRANSFORMER_METRICS_PATH = "transformer_metrics.json"


class ReviewDataset(Dataset):
    """PyTorch dataset for tokenized review texts."""

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: AutoTokenizer,
        max_length: int = 128,
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    """Compute metrics for HuggingFace Trainer."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, average="weighted")),
        "recall": float(recall_score(labels, predictions, average="weighted")),
        "f1": float(f1_score(labels, predictions, average="weighted")),
    }


def train_transformer_model(
    df: Optional[pd.DataFrame] = None,
    test_size: float = 0.2,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Fine-tune DistilBERT on review sentiment data."""
    if df is None:
        df = prepare_dataset()

    df = build_combined_text_column(df)
    train_df, test_df = train_test_split_dataframe(df, test_size=test_size)

    label_map = {"Negative": 0, "Positive": 1}
    id2label = {v: k for k, v in label_map.items()}

    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        TRANSFORMER_MODEL_NAME,
        num_labels=2,
        id2label=id2label,
        label2id=label_map,
    )

    max_length = int(os.getenv("MAX_SEQ_LENGTH", "128"))
    train_dataset = ReviewDataset(
        train_df["combined_text"].tolist(),
        train_df["sentiment"].map(label_map).tolist(),
        tokenizer,
        max_length=max_length,
    )
    test_dataset = ReviewDataset(
        test_df["combined_text"].tolist(),
        test_df["sentiment"].map(label_map).tolist(),
        tokenizer,
        max_length=max_length,
    )

    model_dir = get_model_dir()
    output_dir = model_dir / TRANSFORMER_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    num_epochs = epochs or int(os.getenv("TRANSFORMER_EPOCHS", "2"))
    train_batch_size = batch_size or int(os.getenv("TRANSFORMER_BATCH_SIZE", "16"))

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=train_batch_size,
        warmup_steps=100,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none",
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting DistilBERT fine-tuning for %s epochs...", num_epochs)
    trainer.train()
    eval_results = trainer.evaluate()

    predictions_output = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions_output.predictions, axis=-1)
    y_true = test_df["sentiment"].map(label_map).values

    metrics: Dict[str, Any] = {
        "model": TRANSFORMER_MODEL_NAME,
        "accuracy": float(eval_results.get("eval_accuracy", accuracy_score(y_true, y_pred))),
        "precision": float(eval_results.get("eval_precision", 0)),
        "recall": float(eval_results.get("eval_recall", 0)),
        "f1_score": float(eval_results.get("eval_f1", 0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=["Negative", "Positive"], output_dict=True
        ),
        "label_map": label_map,
    }

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    label_map_path = output_dir / "label_map.json"
    with open(label_map_path, "w", encoding="utf-8") as file:
        json.dump(label_map, file, indent=2)

    metrics_path = model_dir / TRANSFORMER_METRICS_PATH
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    logger.info(
        "Transformer training complete | f1=%.4f | saved to %s",
        metrics["f1_score"],
        output_dir,
    )
    return str(output_dir), metrics


def main() -> None:
    """CLI entry point for transformer training."""
    try:
        output_dir, metrics = train_transformer_model()
        logger.info("Model saved to %s | f1=%.4f", output_dir, metrics["f1_score"])
    except Exception as exc:
        logger.exception("Transformer training failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
