"""Shared utilities for paths, logging, and environment configuration."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "reviews.csv"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_LOG_LEVEL = "INFO"


def get_project_root() -> Path:
    """Return the project root directory."""
    return PROJECT_ROOT


def get_data_dir() -> Path:
    """Return the data directory, creating it if needed."""
    data_dir = PROJECT_ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def get_model_dir() -> Path:
    """Return the models directory, creating it if needed."""
    raw_path = Path(os.getenv("MODEL_DIR", str(DEFAULT_MODEL_DIR)))
    model_dir = raw_path if raw_path.is_absolute() else PROJECT_ROOT / raw_path
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def get_data_path() -> Path:
    """Return the dataset path from environment or default."""
    raw_path = Path(os.getenv("DATA_PATH", str(DEFAULT_DATA_PATH)))
    if not raw_path.is_absolute():
        return PROJECT_ROOT / raw_path
    return raw_path


def load_env(env_file: Optional[Path] = None) -> None:
    """Load environment variables from a .env file."""
    env_path = env_file or PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()


def setup_logging(
    name: str = "amazon_review_intelligence",
    level: Optional[str] = None,
) -> logging.Logger:
    """Configure and return a module logger."""
    log_level = level or os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL)
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.propagate = False
    return logger


def add_src_to_path() -> None:
    """Ensure src directory is on sys.path for imports."""
    src_path = str(PROJECT_ROOT / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
