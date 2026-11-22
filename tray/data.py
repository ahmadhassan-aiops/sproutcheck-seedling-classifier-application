"""Cached loaders: the classifier, its evaluation and the sample photos."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from sprout.model import Classifier

ROOT = Path(__file__).resolve().parents[1]


@st.cache_resource
def classifier() -> Classifier:
    return Classifier()


def meta() -> dict:
    return classifier().meta


@st.cache_data
def predictions() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / "test_predictions.csv")
    df["right"] = df["label"] == df["pred"]
    return df


@st.cache_data
def samples() -> pd.DataFrame:
    return pd.DataFrame(meta()["samples"])


def sample_image(file: str) -> Image.Image:
    return Image.open(ROOT / "samples" / file)


def data_uri(image: Image.Image, size: int = 256) -> str:
    image = image.convert("RGB")
    image.thumbnail((size, size))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
