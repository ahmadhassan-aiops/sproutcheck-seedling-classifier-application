"""Trains and compares seedling classifiers, picks one by validation macro F1, scores the test set once, calibrates the
confidence and exports the model to ONNX for a CPU-only app.

    python train.py --data "<folder with images.npy and Labels.csv>"

Candidates: frozen MobileNetV3-Small features + logistic regression (baseline), and fine-tuned MobileNetV3-Large and
EfficientNet-B0 (ImageNet weights from torchvision). Writes model/, data/ and samples/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.transforms import v2 as T

ROOT = Path(__file__).resolve().parent
SEED, SIZE, EPOCHS, BATCH = 42, 160, 14, 32
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
CROPS = {"Maize", "Common wheat", "Sugar beet"}
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(max(1, torch.get_num_threads()))


def load(folder: Path) -> tuple[np.ndarray, np.ndarray]:
    """Images as RGB (the array is stored BGR), labels; exact duplicate images kept once."""
    x = np.load(folder / "images.npy")[..., ::-1].copy()
    y = pd.read_csv(folder / "Labels.csv")["Label"].to_numpy()
    seen, keep = set(), []
    for i in range(len(x)):
        h = hashlib.md5(x[i].tobytes()).hexdigest()
        if h not in seen:
            seen.add(h)
            keep.append(i)
    return x[keep], y[keep]


class Seedlings(Dataset):
    def __init__(self, x, y, train: bool):
        self.x, self.y = x, y
        aug = [T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(180),
               T.RandomResizedCrop(SIZE, scale=(0.75, 1.0)), T.ColorJitter(0.2, 0.2, 0.15)] if train else [T.Resize(SIZE)]
        self.tf = T.Compose([T.ToImage(), *aug, T.ToDtype(torch.float32, scale=True), T.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.tf(Image.fromarray(self.x[i])), self.y[i]


def build(name: str, n: int) -> nn.Module:
    if name == "mobilenet_v3_large":
        m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, n)
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n)
    else:
        raise ValueError(name)
    return m


@torch.no_grad()
def logits_of(model: nn.Module, loader: DataLoader) -> np.ndarray:
    model.eval()
    return torch.cat([model(xb) for xb, _ in loader]).numpy()


def scores(y, pred, classes) -> dict:
    return {"accuracy": round(float(accuracy_score(y, pred)), 4),
            "macro_f1": round(float(f1_score(y, pred, average="macro")), 4),
            "recall": {c: round(float(r), 3) for c, r in zip(classes, recall_score(y, pred, average=None, labels=range(len(classes))))}}


def fine_tune(name, train_dl, val_dl, yv, classes, log) -> tuple[nn.Module, dict, list]:
    model = build(name, len(classes))
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=EPOCHS * len(train_dl), pct_start=0.2)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    best, best_state, history = -1.0, None, []
    for ep in range(1, EPOCHS + 1):
        model.train()
        t0, total = time.time(), 0.0
        for xb, yb in train_dl:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(yb)
        pv = logits_of(model, val_dl).argmax(1)
        s = scores(yv, pv, classes)
        history.append({"epoch": ep, "loss": round(total / len(train_dl.dataset), 4), "val_accuracy": s["accuracy"],
                        "val_macro_f1": s["macro_f1"], "seconds": round(time.time() - t0, 1)})
        log(f"{name} epoch {ep}: {history[-1]}")
        if s["macro_f1"] > best:
            best, best_state = s["macro_f1"], {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, scores(yv, logits_of(model, val_dl).argmax(1), classes), history


def temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """One scalar that makes the softmax confidence match how often it is right on the validation set."""
    lt, yt = torch.tensor(logits), torch.tensor(y)
    t = torch.ones(1, requires_grad=True)
    opt = torch.optim.LBFGS([t], lr=0.1, max_iter=200)

    def step():
        opt.zero_grad()
        loss = nn.functional.cross_entropy(lt / t.clamp(min=0.05), yt)
        loss.backward()
        return loss
    opt.step(step)
    return float(t.clamp(min=0.05).item())


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    args = ap.parse_args()
    logf = open(ROOT / "train_log.txt", "w", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    x, labels = load(args.data)
    classes = sorted(set(labels))
    y = np.array([classes.index(l) for l in labels])
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.15, stratify=y, random_state=SEED)
    tr, va = train_test_split(tr, test_size=0.15 / 0.85, stratify=y[tr], random_state=SEED)
    log(f"images {len(y)} (after duplicates) | train {len(tr)} val {len(va)} test {len(te)}")
    train_dl = DataLoader(Seedlings(x[tr], y[tr], True), batch_size=BATCH, shuffle=True)
    val_dl = DataLoader(Seedlings(x[va], y[va], False), batch_size=64)
    test_dl = DataLoader(Seedlings(x[te], y[te], False), batch_size=64)

    # Baseline: frozen ImageNet features + logistic regression
    feat = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    feat.classifier = nn.Identity()
    plain_tr = DataLoader(Seedlings(x[tr], y[tr], False), batch_size=64)
    ftr, fva = logits_of(feat, plain_tr), logits_of(feat, val_dl)
    lr = LogisticRegression(max_iter=3000, C=1.0).fit(ftr, y[tr])
    results = {"frozen_features_logreg": {"val": scores(y[va], lr.predict(fva), classes)}}
    log(f"baseline val: {results['frozen_features_logreg']['val']['accuracy']} / {results['frozen_features_logreg']['val']['macro_f1']}")

    trained = {}
    for name in ["mobilenet_v3_large", "efficientnet_b0"]:
        model, val, hist = fine_tune(name, train_dl, val_dl, y[va], classes, log)
        trained[name] = model
        results[name] = {"val": val, "history": hist, "parameters": sum(p.numel() for p in model.parameters())}

    chosen = max(trained, key=lambda m: (results[m]["val"]["macro_f1"], results[m]["val"]["accuracy"]))
    model = trained[chosen]
    log(f"chosen by validation macro F1: {chosen}")
    temp = temperature(logits_of(model, val_dl), y[va])
    test_logits = logits_of(model, test_dl)
    test_prob = softmax(test_logits / temp)
    test_pred = test_prob.argmax(1)
    test = scores(y[te], test_pred, classes)
    results["frozen_features_logreg"]["test"] = scores(y[te], lr.predict(logits_of(feat, test_dl)), classes)
    log(f"test: {test['accuracy']} / {test['macro_f1']} | temperature {temp:.3f}")

    # Export to ONNX and check it gives the same answers
    (ROOT / "model").mkdir(exist_ok=True)
    onnx_path = ROOT / "model" / "sproutcheck.onnx"
    model.eval()
    torch.onnx.export(model, torch.randn(1, 3, SIZE, SIZE), onnx_path, input_names=["image"], output_names=["logits"],
                      dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17, dynamo=False)
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    xb = next(iter(test_dl))[0].numpy()
    diff = float(np.abs(sess.run(None, {"image": xb})[0] - test_logits[: len(xb)]).max())
    log(f"onnx max logit difference: {diff:.2e}")

    # Calibration on the test set: confidence bands
    conf = test_prob.max(1)
    bands = []
    for lo, hi in [(0, .6), (.6, .8), (.8, .95), (.95, 1.01)]:
        m = (conf >= lo) & (conf < hi)
        bands.append({"band": f"{int(lo * 100)}-{min(int(hi * 100), 100)}%", "n": int(m.sum()),
                      "mean_confidence": round(float(conf[m].mean()), 3) if m.any() else None,
                      "accuracy": round(float((test_pred[m] == y[te][m]).mean()), 3) if m.any() else None})

    # Ship the test predictions and a few test images per species (never training images)
    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "samples").mkdir(exist_ok=True)
    rows = []
    for k, i in enumerate(te):
        rows.append({"image_id": int(i), "label": classes[y[i]], "pred": classes[test_pred[k]],
                     "confidence": round(float(conf[k]), 4),
                     **{f"p_{c}": round(float(test_prob[k, j]), 4) for j, c in enumerate(classes)}})
    pd.DataFrame(rows).to_csv(ROOT / "data" / "test_predictions.csv", index=False)
    samples = []
    for c in range(len(classes)):
        ids = [i for i in te if y[i] == c][:6]
        for i in ids:
            name = f"{classes[c].replace(' ', '_').replace('-', '_').lower()}_{i}.png"
            Image.fromarray(x[i]).save(ROOT / "samples" / name, optimize=True)
            samples.append({"file": name, "image_id": int(i), "label": classes[c]})

    meta = {
        "classes": classes, "crops": sorted(CROPS), "input_size": SIZE, "mean": MEAN, "std": STD,
        "temperature": round(temp, 4), "chosen": chosen, "selection": "highest validation macro F1, then accuracy",
        "counts": {"images": int(len(y)), "duplicates_removed": 4750 - int(len(y)), "train": int(len(tr)),
                   "val": int(len(va)), "test": int(len(te)),
                   "per_class": {c: int((y == j).sum()) for j, c in enumerate(classes)}},
        "results": results, "test": test,
        "confusion": confusion_matrix(y[te], test_pred, labels=range(len(classes))).tolist(),
        "calibration": bands, "onnx_max_diff": diff, "onnx_mb": round(onnx_path.stat().st_size / 1e6, 2),
        "samples": samples, "epochs": EPOCHS,
    }
    (ROOT / "model" / "metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    log("done")


if __name__ == "__main__":
    main()
