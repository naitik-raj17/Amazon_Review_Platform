"""Generate sample Amazon review dataset for development and testing."""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from utils import get_data_dir, setup_logging

logger = setup_logging(__name__)

COUNTRIES = ["United States", "United Kingdom", "India", "Canada", "Germany", "Australia"]
REVIEWER_NAMES = [
    "Alice Johnson", "Bob Smith", "Carol White", "David Lee", "Emma Brown",
    "Frank Miller", "Grace Taylor", "Henry Wilson", "Ivy Martinez", "Jack Anderson",
]

POSITIVE_TITLES = [
    "Excellent product", "Great value", "Highly recommend", "Love it",
    "Perfect purchase", "Amazing quality", "Best buy ever",
]
NEGATIVE_TITLES = [
    "Disappointed", "Poor quality", "Not worth it", "Terrible experience",
    "Waste of money", "Very bad", "Do not buy",
]
NEUTRAL_TITLES = ["It's okay", "Average product", "Mixed feelings"]

POSITIVE_TEXTS = [
    "The product quality is outstanding and exceeded my expectations.",
    "Fast delivery and excellent packaging. Very happy with this purchase.",
    "Great customer service helped resolve my query quickly.",
    "Amazing value for money. The build quality is top notch.",
    "Arrived on time and works perfectly. Highly recommend to everyone.",
]
NEGATIVE_TEXTS = [
    "Product arrived late and packaging was damaged.",
    "Poor quality material. Broke after one week of use.",
    "Customer service was unhelpful and refused a refund.",
    "Overpriced for what you get. Not worth the money at all.",
    "Shipping took forever and the box was completely crushed.",
]
NEUTRAL_TEXTS = [
    "Product is average. Nothing special but does the job.",
    "Delivery was fine but product quality could be better.",
    "Decent price but expected more based on the reviews.",
]


def generate_sample_reviews(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic Amazon reviews dataset."""
    random.seed(seed)
    records = []

    for i in range(n):
        rating = random.choices(
            [1, 2, 3, 4, 5],
            weights=[0.1, 0.15, 0.15, 0.3, 0.3],
        )[0]

        if rating >= 4:
            title = random.choice(POSITIVE_TITLES)
            text = random.choice(POSITIVE_TEXTS)
        elif rating <= 2:
            title = random.choice(NEGATIVE_TITLES)
            text = random.choice(NEGATIVE_TEXTS)
        else:
            title = random.choice(NEUTRAL_TITLES)
            text = random.choice(NEUTRAL_TEXTS)

        review_date = datetime(2023, 1, 1) + timedelta(days=random.randint(0, 730))
        experience_date = review_date - timedelta(days=random.randint(1, 30))

        records.append({
            "Reviewer Name": random.choice(REVIEWER_NAMES),
            "Profile Link": f"https://amazon.com/gp/profile/amzn1.account.{i}",
            "Country": random.choice(COUNTRIES),
            "Review Count": random.randint(1, 200),
            "Review Date": review_date.strftime("%Y-%m-%d"),
            "Rating": rating,
            "Review Title": f"{title} #{i}",
            "Review Text": f"{text} (review id {i})",
            "Date of Experience": experience_date.strftime("%Y-%m-%d"),
        })

    return pd.DataFrame(records)


def main() -> None:
    """Write sample dataset to data/reviews.csv."""
    data_dir = get_data_dir()
    output_path = data_dir / "reviews.csv"
    df = generate_sample_reviews()
    df.to_csv(output_path, index=False)
    logger.info("Generated %s sample reviews at %s", len(df), output_path)


if __name__ == "__main__":
    main()
