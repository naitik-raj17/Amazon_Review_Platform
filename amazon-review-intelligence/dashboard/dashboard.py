"""Streamlit dashboard for Amazon Review Intelligence Platform."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

from data_loader import prepare_dataset  # noqa: E402
from eda import (  # noqa: E402
    get_dataset_statistics,
    plot_country_ratings,
    plot_review_trends,
    plot_sentiment_distribution,
)
from predict import SentimentPredictor  # noqa: E402
from preprocess import TextPreprocessor, build_combined_text_column  # noqa: E402
from utils import get_model_dir, load_env, setup_logging  # noqa: E402

load_env()
logger = setup_logging("dashboard")

st.set_page_config(
    page_title="Amazon Review Intelligence",
    page_icon="📊",
    layout="wide",
)


@st.cache_data(show_spinner="Loading dataset...")
def load_data() -> pd.DataFrame:
    """Load and cache the prepared dataset."""
    try:
        return prepare_dataset()
    except FileNotFoundError:
        return pd.DataFrame()


@st.cache_resource
def load_predictor() -> SentimentPredictor:
    """Load and cache the sentiment predictor."""
    model_type = os.getenv("DEFAULT_MODEL_TYPE", "transformer")
    return SentimentPredictor(model_type=model_type)  # type: ignore[arg-type]


def load_topic_report() -> dict:
    """Load topic modeling report if available."""
    report_path = get_model_dir() / "topic_report.json"
    if report_path.exists():
        with open(report_path, encoding="utf-8") as file:
            return json.load(file)
    return {}


def render_header() -> None:
    st.title("Amazon Customer Review Intelligence Platform")
    st.markdown(
        "Analyze Amazon customer reviews with sentiment classification, "
        "topic modeling, and real-time prediction."
    )


def render_statistics(df: pd.DataFrame) -> None:
    st.subheader("Dataset Statistics")
    if df.empty:
        st.warning("No dataset found. Place `reviews.csv` in the `data/` directory.")
        return

    stats = get_dataset_statistics(df)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Reviews", stats["total_reviews"])
    col2.metric("Countries", stats["unique_countries"])
    col3.metric("Average Rating", stats["average_rating"])
    col4.metric("Median Rating", stats["median_rating"])


def render_sentiment_distribution(df: pd.DataFrame) -> None:
    st.subheader("Sentiment Distribution")
    if "sentiment" not in df.columns:
        st.info("Sentiment labels not available.")
        return
    fig = plot_sentiment_distribution(df)
    st.plotly_chart(fig, use_container_width=True)


def render_country_analytics(df: pd.DataFrame) -> None:
    st.subheader("Country Analytics")
    if "Country" not in df.columns:
        st.info("Country data not available.")
        return
    fig = plot_country_ratings(df)
    st.plotly_chart(fig, use_container_width=True)


def render_review_trends(df: pd.DataFrame) -> None:
    st.subheader("Review Trends Over Time")
    try:
        fig = plot_review_trends(df)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:
        st.warning(f"Could not render trends: {exc}")


def render_topic_analytics() -> None:
    st.subheader("Topic Analytics")
    report = load_topic_report()
    if not report:
        st.info("Topic report not found. Run `python src/topic_modeling.py` first.")
        return

    theme_summary = report.get("theme_summary", {})
    theme_data = [
        {"theme": theme, "topic_count": len(entries)}
        for theme, entries in theme_summary.items()
        if entries
    ]

    if theme_data:
        theme_df = pd.DataFrame(theme_data)
        fig = px.bar(
            theme_df,
            x="theme",
            y="topic_count",
            title="Topics Mapped to Customer Themes",
            color="topic_count",
            color_continuous_scale="Teal",
        )
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Detailed Topic Report"):
        st.json(report)


def render_live_prediction() -> None:
    st.subheader("Live Sentiment Prediction")
    review_text = st.text_area(
        "Enter a review",
        placeholder="Product arrived late and packaging was damaged",
        height=120,
    )

    if st.button("Predict Sentiment", type="primary"):
        if not review_text.strip():
            st.error("Please enter a review text.")
            return

        try:
            predictor = load_predictor()
            result = predictor.predict(review_text)
            sentiment = result["sentiment"]
            confidence = result["confidence"]

            color = "#2ecc71" if sentiment == "Positive" else "#e74c3c"
            st.markdown(
                f"""
                <div style="padding: 20px; border-radius: 10px; background-color: {color}22;
                            border-left: 5px solid {color};">
                    <h3 style="margin:0; color: {color};">{sentiment}</h3>
                    <p style="margin:5px 0 0 0;">Confidence: <b>{confidence:.2%}</b></p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        except FileNotFoundError:
            st.error(
                "No trained model found. Run training scripts first:\n"
                "`python src/train_tfidf.py` or `python src/train_transformer.py`"
            )
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


def main() -> None:
    render_header()
    df = load_data()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Overview", "Sentiment", "Geography", "Topics", "Predict"]
    )

    with tab1:
        render_statistics(df)
        if not df.empty:
            render_review_trends(df)

    with tab2:
        render_sentiment_distribution(df)

    with tab3:
        render_country_analytics(df)

    with tab4:
        render_topic_analytics()

    with tab5:
        render_live_prediction()


if __name__ == "__main__":
    main()
