"""Reusable NLP preprocessing pipeline for review text."""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Union

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from utils import setup_logging

logger = setup_logging(__name__)

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
HTML_PATTERN = re.compile(r"<[^>]+>")
PUNCTUATION_PATTERN = re.compile(r"[^\w\s]")
NUMBER_PATTERN = re.compile(r"\d+")


def ensure_nltk_resources() -> None:
    """Download required NLTK resources if not present."""
    resources = ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]
    for resource in resources:
        try:
            if resource == "punkt":
                nltk.data.find("tokenizers/punkt")
            elif resource == "punkt_tab":
                nltk.data.find("tokenizers/punkt_tab")
            elif resource == "stopwords":
                nltk.data.find("corpora/stopwords")
            elif resource in ("wordnet", "omw-1.4"):
                nltk.data.find(f"corpora/{resource}")
        except LookupError:
            logger.info("Downloading NLTK resource: %s", resource)
            nltk.download(resource, quiet=True)


class TextPreprocessor:
    """Pipeline for cleaning and normalizing review text."""

    def __init__(self, language: str = "english") -> None:
        ensure_nltk_resources()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words(language))

    def combine_title_and_text(
        self,
        title: Union[str, float, None],
        text: Union[str, float, None],
    ) -> str:
        """Combine review title and body into a single text field."""
        title_str = "" if pd.isna(title) else str(title).strip()
        text_str = "" if pd.isna(text) else str(text).strip()
        return f"{title_str} {text_str}".strip()

    def clean_text(self, text: str) -> str:
        """Apply full preprocessing pipeline to a single text string."""
        if not text or not isinstance(text, str):
            return ""

        cleaned = text.lower()
        cleaned = URL_PATTERN.sub(" ", cleaned)
        cleaned = HTML_PATTERN.sub(" ", cleaned)
        cleaned = PUNCTUATION_PATTERN.sub(" ", cleaned)
        cleaned = NUMBER_PATTERN.sub(" ", cleaned)
        tokens = cleaned.split()
        tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words and len(token) > 1
        ]
        return " ".join(tokens)

    def transform(self, texts: Iterable[str]) -> List[str]:
        """Preprocess an iterable of text strings."""
        return [self.clean_text(text) for text in texts]

    def transform_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "combined_text",
        output_column: str = "processed_text",
    ) -> pd.DataFrame:
        """Add a processed text column to a dataframe."""
        result = df.copy()
        if text_column not in result.columns:
            result[text_column] = result.apply(
                lambda row: self.combine_title_and_text(
                    row.get("Review Title"), row.get("Review Text")
                ),
                axis=1,
            )
        result[output_column] = self.transform(result[text_column].astype(str))
        return result


def build_combined_text_column(df: pd.DataFrame) -> pd.DataFrame:
    """Create combined_text column from title and review body."""
    preprocessor = TextPreprocessor()
    result = df.copy()
    result["combined_text"] = result.apply(
        lambda row: preprocessor.combine_title_and_text(
            row.get("Review Title"), row.get("Review Text")
        ),
        axis=1,
    )
    return result
