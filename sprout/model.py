"""Seedling classifier for the app: an ONNX model run on CPU with onnxruntime, no deep learning framework needed."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 8_000_000


class Classifier:
    def __init__(self, folder: Path = ROOT / "model"):
        import onnxruntime as ort

        self.meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        self.session = ort.InferenceSession(str(folder / "sproutcheck.onnx"), providers=["CPUExecutionProvider"])
        self.classes: list[str] = self.meta["classes"]
        self.crops: set[str] = set(self.meta["crops"])
        self.size: int = self.meta["input_size"]
        self.mean = np.array(self.meta["mean"], dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(self.meta["std"], dtype=np.float32).reshape(3, 1, 1)
        self.temperature: float = self.meta["temperature"]

    def prepare(self, image: Image.Image) -> np.ndarray:
        """Square centre crop, resize to the training size, scale and normalise like the test images."""
        image = ImageOps.exif_transpose(image).convert("RGB")
        side = min(image.size)
        image = ImageOps.fit(image, (side, side), centering=(0.5, 0.5)).resize((self.size, self.size), Image.BILINEAR)
        arr = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        return ((arr - self.mean) / self.std)[None].astype(np.float32)

    def probabilities(self, image: Image.Image) -> dict[str, float]:
        logits = self.session.run(None, {"image": self.prepare(image)})[0][0] / self.temperature
        e = np.exp(logits - logits.max())
        p = e / e.sum()
        return dict(sorted(zip(self.classes, map(float, p)), key=lambda kv: -kv[1]))


def open_image(data: bytes) -> Image.Image:
    """An uploaded photo, or ValueError with a readable message."""
    if len(data) > MAX_BYTES:
        raise ValueError("The photo is larger than 8 MB")
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError) as err:
        raise ValueError("This file is not an image SproutCheck can read") from err
    if min(image.size) < 64:
        raise ValueError("The photo is too small (at least 64 pixels on each side)")
    return image
