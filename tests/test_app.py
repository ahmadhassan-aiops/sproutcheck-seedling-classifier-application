"""The shipped model reproduces its evaluation on the sample photos; uploads are checked; every page renders."""

from __future__ import annotations

import io

import pandas as pd
import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

from sprout.model import Classifier, open_image

PAGES = [("views/identify.py", "render"), ("views/tray.py", "render"), ("views/accuracy.py", "render"),
         ("views/method.py", "species_page"), ("views/method.py", "render")]


@pytest.fixture(scope="module")
def clf():
    return Classifier()


def test_model_matches_saved_test_predictions(clf):
    preds = pd.read_csv("data/test_predictions.csv").set_index("image_id")
    for s in clf.meta["samples"]:
        probs = clf.probabilities(Image.open(f"samples/{s['file']}"))
        top, p = next(iter(probs.items()))
        row = preds.loc[s["image_id"]]
        assert top == row["pred"], s["file"]
        assert abs(p - row["confidence"]) < 0.02, s["file"]


def test_probabilities_sum_to_one_and_cover_every_species(clf):
    probs = clf.probabilities(Image.new("RGB", (300, 200), (40, 30, 20)))
    assert set(probs) == set(clf.classes) and abs(sum(probs.values()) - 1) < 1e-4


def test_uploads_are_checked():
    with pytest.raises(ValueError):
        open_image(b"not an image")
    small = io.BytesIO()
    Image.new("RGB", (20, 20)).save(small, format="PNG")
    with pytest.raises(ValueError):
        open_image(small.getvalue())


def run(module: str, fn: str) -> AppTest:
    script = f"""
import importlib
from tray.ui import inject_css
inject_css()
getattr(importlib.import_module("{module.replace('/', '.').removesuffix('.py')}"), "{fn}")()
"""
    return AppTest.from_string(script, default_timeout=60).run()


@pytest.mark.parametrize("page,fn", PAGES)
def test_page_renders(page, fn):
    at = run(page, fn)
    assert not at.exception, at.exception


def test_identify_species_and_upload_mode():
    at = run("views/identify.py", "render")
    at.selectbox(key="i_species").set_value("Maize").run()
    assert not at.exception
    at.segmented_control(key="i_src").set_value("Upload").run()
    assert not at.exception and at.info
