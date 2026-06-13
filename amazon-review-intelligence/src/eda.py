"""Exploratory data analysis and visualization functions."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns

from utils import get_data_dir, setup_logging

logger = setup_logging(__name__)

sns.set_theme(style="whitegrid")


def plot_rating_distribution(
    df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot histogram of review ratings."""
    fig, ax = plt.subplots(figsize=(8, 5))
    rating_counts = df["Rating"].value_counts().sort_index()
    sns.barplot(x=rating_counts.index, y=rating_counts.values, ax=ax, palette="viridis")
    ax.set_title("Rating Distribution")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Count")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved rating distribution plot to %s", save_path)
    return fig


def plot_country_ratings(df: pd.DataFrame) -> plotly.graph_objects.Figure:
    """Create interactive country-wise average rating chart."""
    country_stats = (
        df.groupby("Country")["Rating"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "avg_rating", "count": "review_count"})
        .sort_values("review_count", ascending=False)
        .head(15)
    )
    fig = px.bar(
        country_stats,
        x="Country",
        y="avg_rating",
        color="review_count",
        title="Average Rating by Country (Top 15)",
        labels={"avg_rating": "Average Rating", "review_count": "Review Count"},
        color_continuous_scale="Blues",
    )
    fig.update_layout(xaxis_tickangle=-45)
    return fig


def plot_review_trends(df: pd.DataFrame) -> plotly.graph_objects.Figure:
    """Plot review volume trends over time."""
    trend_df = df.copy()
    trend_df["Review Date"] = pd.to_datetime(trend_df["Review Date"], errors="coerce")
    trend_df = trend_df.dropna(subset=["Review Date"])
    trend_df["month"] = trend_df["Review Date"].dt.to_period("M").astype(str)

    monthly = trend_df.groupby("month").size().reset_index(name="review_count")
    fig = px.line(
        monthly,
        x="month",
        y="review_count",
        title="Review Trends Over Time",
        markers=True,
    )
    fig.update_layout(xaxis_tickangle=-45)
    return fig


def plot_word_frequency(
    texts: pd.Series,
    top_n: int = 20,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot top word frequencies from processed text."""
    word_counts: dict[str, int] = {}
    for text in texts.dropna():
        for word in str(text).split():
            word_counts[word] = word_counts.get(word, 0) + 1

    top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    words, counts = zip(*top_words) if top_words else ([], [])

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=list(counts), y=list(words), ax=ax, palette="rocket")
    ax.set_title(f"Top {top_n} Word Frequencies")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Word")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_review_length_distribution(
    df: pd.DataFrame,
    text_column: str = "combined_text",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot distribution of review text lengths."""
    lengths = df[text_column].astype(str).str.len()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(lengths, bins=50, kde=True, ax=ax, color="steelblue")
    ax.set_title("Review Length Distribution")
    ax.set_xlabel("Character Count")
    ax.set_ylabel("Frequency")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_sentiment_distribution(df: pd.DataFrame) -> plotly.graph_objects.Figure:
    """Plot sentiment label distribution."""
    if "sentiment" not in df.columns:
        raise ValueError("DataFrame must contain a 'sentiment' column.")

    counts = df["sentiment"].value_counts().reset_index()
    counts.columns = ["sentiment", "count"]
    fig = px.pie(
        counts,
        names="sentiment",
        values="count",
        title="Sentiment Distribution",
        color="sentiment",
        color_discrete_map={"Positive": "#2ecc71", "Negative": "#e74c3c"},
    )
    return fig


def get_dataset_statistics(df: pd.DataFrame) -> dict:
    """Return summary statistics for the dataset."""
    stats = {
        "total_reviews": len(df),
        "unique_countries": df["Country"].nunique() if "Country" in df.columns else 0,
        "average_rating": round(float(df["Rating"].mean()), 2),
        "median_rating": float(df["Rating"].median()),
        "date_range": {
            "min": str(pd.to_datetime(df["Review Date"], errors="coerce").min()),
            "max": str(pd.to_datetime(df["Review Date"], errors="coerce").max()),
        },
    }
    if "sentiment" in df.columns:
        stats["sentiment_counts"] = df["sentiment"].value_counts().to_dict()
    return stats


def save_eda_plots(df: pd.DataFrame, output_dir: Optional[Path] = None) -> None:
    """Generate and save all EDA plots."""
    out_dir = output_dir or get_data_dir() / "eda_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_rating_distribution(df, save_path=out_dir / "rating_distribution.png")
    plt.close("all")
    logger.info("EDA plots saved to %s", out_dir)
