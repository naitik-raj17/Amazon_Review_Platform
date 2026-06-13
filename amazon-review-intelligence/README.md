# Amazon Customer Review Intelligence Platform

An end-to-end NLP application that analyzes Amazon customer reviews and provides **sentiment classification**, **topic extraction**, **analytics**, and **real-time prediction**.

## Features

- **Data Pipeline** — Safe CSV loading, missing value handling, deduplication, and text validation
- **NLP Preprocessing** — Reusable pipeline with lemmatization, stopword removal, and text normalization
- **Sentiment Classification**
  - Traditional ML: TF-IDF + Logistic Regression / Random Forest
  - Deep Learning: Fine-tuned DistilBERT (`distilbert-base-uncased`)
- **Topic Modeling** — BERTopic with theme mapping (Delivery, Packaging, Product Quality, etc.)
- **FastAPI Backend** — REST API for real-time sentiment prediction
- **Streamlit Dashboard** — Interactive analytics and live prediction interface

## Project Structure

```
amazon-review-intelligence/
├── data/                   # Dataset directory
├── notebooks/              # Jupyter notebooks for EDA and modeling
├── src/                    # Core Python modules
├── api/                    # FastAPI application
├── dashboard/              # Streamlit dashboard
├── models/                 # Saved model artifacts
├── requirements.txt
├── README.md
└── .env.example
```

## Prerequisites

- Python 3.11+
- 8 GB RAM recommended (transformer training)
- GPU optional (speeds up transformer training)

## Setup

### 1. Clone and enter the project

```bash
cd amazon-review-intelligence
```

### 2. Create a virtual environment

```bash
python3.11 -m venv venv
source venv/bin/activate   # Linux/macOS
# venv\Scripts\activate    # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

```bash
cp .env.example .env
```

Edit `.env` to customize paths, model settings, and API configuration.

### 5. Prepare dataset

Place your Amazon reviews CSV at `data/reviews.csv` with these columns:

| Column | Description |
|--------|-------------|
| Reviewer Name | Reviewer display name |
| Profile Link | Amazon profile URL |
| Country | Reviewer country |
| Review Count | Total reviews by reviewer |
| Review Date | Date review was posted |
| Rating | Star rating (1–5) |
| Review Title | Review headline |
| Review Text | Full review body |
| Date of Experience | Date of product experience |

**Or generate sample data for testing:**

```bash
python src/generate_sample_data.py
```

## Usage

### Run Exploratory Analysis (Notebooks)

```bash
jupyter notebook notebooks/
```

Notebooks:
1. `01_eda.ipynb` — Exploratory data analysis
2. `02_preprocessing.ipynb` — Text preprocessing pipeline
3. `03_tfidf_model.ipynb` — TF-IDF model training
4. `04_transformer_model.ipynb` — DistilBERT fine-tuning

### Train Models

```bash
# TF-IDF + Logistic Regression / Random Forest
python src/train_tfidf.py

# DistilBERT fine-tuning
python src/train_transformer.py

# BERTopic topic modeling
python src/topic_modeling.py
```

Trained models are saved under `models/`.

### Start the API

```bash
cd api
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Or from the project root:

```bash
python api/app.py
```

**API Documentation:** http://localhost:8000/docs

#### API Examples

**Health check:**

```bash
curl http://localhost:8000/
```

Response:

```json
{
  "status": "ok",
  "message": "Amazon Customer Review Intelligence Platform API is running",
  "model_type": "transformer"
}
```

**Predict sentiment:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"review": "Product arrived late and packaging was damaged"}'
```

Response:

```json
{
  "sentiment": "Negative",
  "confidence": 0.97
}
```

**Python client example:**

```python
import httpx

response = httpx.post(
    "http://localhost:8000/predict",
    json={"review": "Amazing product quality and fast delivery!"},
)
print(response.json())
# {'sentiment': 'Positive', 'confidence': 0.98}
```

### Launch the Dashboard

```bash
streamlit run dashboard/dashboard.py
```

Open http://localhost:8501 in your browser.

## Sentiment Label Rules

| Rating | Label |
|--------|-------|
| ≥ 4 | Positive |
| ≤ 2 | Negative |
| 3 | Excluded from training |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_PATH` | `data/reviews.csv` | Path to dataset |
| `MODEL_DIR` | `models` | Model output directory |
| `DEFAULT_MODEL_TYPE` | `transformer` | API/dashboard model (`tfidf` or `transformer`) |
| `API_HOST` | `0.0.0.0` | API bind host |
| `API_PORT` | `8000` | API port |
| `TRANSFORMER_EPOCHS` | `2` | DistilBERT training epochs |
| `TRANSFORMER_BATCH_SIZE` | `16` | Training batch size |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

## Module Overview

| Module | Purpose |
|--------|---------|
| `src/data_loader.py` | Dataset loading, cleaning, label creation |
| `src/preprocess.py` | NLP preprocessing pipeline |
| `src/eda.py` | Visualization and statistics functions |
| `src/train_tfidf.py` | TF-IDF model training |
| `src/train_transformer.py` | DistilBERT fine-tuning |
| `src/topic_modeling.py` | BERTopic topic extraction |
| `src/predict.py` | Unified prediction interface |
| `src/utils.py` | Logging, paths, environment config |
| `api/app.py` | FastAPI REST endpoints |
| `dashboard/dashboard.py` | Streamlit analytics dashboard |

## Production Notes

- All modules use **type hints**, **logging**, and **exception handling**
- Paths are configurable via environment variables (no hardcoded paths)
- Models are persisted under `models/` using joblib (TF-IDF) and HuggingFace format (transformer)
- The API falls back to TF-IDF if the transformer model is unavailable

## License

MIT
