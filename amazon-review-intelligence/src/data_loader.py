"""Dataset loading, cleaning, and sentiment label creation."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from utils import get_data_path, setup_logging

logger = setup_logging(__name__)

EXPECTED_COLUMNS = [
    "Reviewer Name",
    "Profile Link",
    "Country",
    "Review Count",
    "Review Date",
    "Rating",
    "Review Title",
    "Review Text",
    "Date of Experience",
]

COLUMN_ALIASES = {
    "reviewer_name": "Reviewer Name",
    "profile_link": "Profile Link",
    "country": "Country",
    "review_count": "Review Count",
    "review_date": "Review Date",
    "rating": "Rating",
    "review_title": "Review Title",
    "review_text": "Review Text",
    "date_of_experience": "Date of Experience",
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to expected schema."""
    renamed = {}
    for col in df.columns:
        normalized = col.strip()
        if normalized in EXPECTED_COLUMNS:
            continue
        key = normalized.lower().replace(" ", "_")
        if key in COLUMN_ALIASES:
            renamed[col] = COLUMN_ALIASES[key]
    if renamed:
        df = df.rename(columns=renamed)
    return df


def load_dataset(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the Amazon reviews CSV dataset safely."""
    data_path = Path(path) if path else get_data_path()

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. "
            "Place reviews.csv in the data/ directory or set DATA_PATH."
        )

    try:
        df = pd.read_csv(
            data_path,
            engine="python",
            on_bad_lines="skip"
        )
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Dataset at {data_path} is empty.") from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to load dataset from {data_path}: {exc}") from exc

    df = _normalize_columns(df)
    logger.info("Loaded %s rows from %s", len(df), data_path)
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill or drop missing values in critical columns."""
    cleaned = df.copy()

    text_cols = ["Review Title", "Review Text"]
    for col in text_cols:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].fillna("")

    if "Country" in cleaned.columns:
        cleaned["Country"] = cleaned["Country"].fillna("Unknown")

    # if "Rating" in cleaned.columns:
    #     cleaned["Rating"] = pd.to_numeric(cleaned["Rating"], errors="coerce")

    if "Rating" in cleaned.columns:
        cleaned["Rating"] = (
            cleaned["Rating"]
            .astype(str)
            .str.extract(r"Rated\s+([1-5])\s+out")[0]
            .astype(float)
        )
        # cleaned["Rating"] = pd.to_numeric(cleaned["Rating"][0], errors="coerce")

        logger.info(
        "Rating distribution:\n%s",
        cleaned["Rating"].value_counts().sort_index()
        )
    before = len(cleaned)
    cleaned = cleaned.dropna(subset=["Rating", "Review Text"])
    dropped = before - len(cleaned)
    if dropped:
        logger.info("Dropped %s rows with missing Rating or Review Text", dropped)

    return cleaned.reset_index(drop=True)


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate review records."""
    subset = [col for col in ["Review Text", "Review Title", "Rating"] if col in df.columns]
    before = len(df)
    deduped = df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
    removed = before - len(deduped)
    if removed:
        logger.info("Removed %s duplicate rows", removed)
    return deduped


def validate_text_records(df: pd.DataFrame, min_length: int = 5) -> pd.DataFrame:
    """Remove rows with invalid or too-short review text."""
    cleaned = df.copy()
    cleaned["combined_text"] = (
        cleaned.get("Review Title", "").astype(str).str.strip()
        + " "
        + cleaned.get("Review Text", "").astype(str).str.strip()
    ).str.strip()

    mask = (
        cleaned["combined_text"].str.len() >= min_length
    ) & (~cleaned["combined_text"].str.lower().isin(["nan", "none", ""]))
    before = len(cleaned)
    cleaned = cleaned[mask].reset_index(drop=True)
    removed = before - len(cleaned)
    if removed:
        logger.info("Removed %s invalid text records", removed)
    return cleaned


def create_sentiment_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create sentiment labels from ratings.

    Rating >= 4 -> Positive
    Rating <= 2 -> Negative
    Rating == 3 -> excluded
    """
    labeled = df.copy()
    labeled["Rating"] = labeled["Rating"].astype(float)

    labeled = labeled[labeled["Rating"] != 3].copy()
    labeled["sentiment"] = labeled["Rating"].apply(
        lambda rating: "Positive" if rating >= 4 else "Negative"
    )
    logger.info(
        "Created sentiment labels: %s positive, %s negative",
        (labeled["sentiment"] == "Positive").sum(),
        (labeled["sentiment"] == "Negative").sum(),
    )
    return labeled.reset_index(drop=True)


def prepare_dataset(path: Optional[Path] = None) -> pd.DataFrame:
    """Run the full dataset preparation pipeline."""
    df = load_dataset(path)
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = validate_text_records(df)
    df = create_sentiment_labels(df)
    logger.info("Prepared dataset with %s training-ready rows", len(df))
    return df


def train_test_split_dataframe(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataframe into train and test sets with stratification."""
    from sklearn.model_selection import train_test_split

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["sentiment"],
    )
    logger.info("Split data: train=%s, test=%s", len(train_df), len(test_df))
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
