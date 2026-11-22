# SproutCheck: seedling photos, crop or weed

**Live app:** [sproutcheck-seedling-classifier-application.streamlit.app](https://sproutcheck-seedling-classifier-application.streamlit.app/) · **Notebook and project folder:** [AIOPS-Portfolio-Projects](https://github.com/ahmadhassan-aiops/AIOPS-Portfolio-Projects/tree/main/4.%20Computer%20Vision%20Projects/5.%20CV%20Project)

Is this seedling a crop to keep or a weed to pull? SproutCheck, by **Ahmad Hassan** (AI Engineer), names a seedling
from a top-down photo among 12 species (maize, common wheat and sugar beet are crops; nine are weeds), gives the keep
or remove verdict, and shows how sure it is and what else it could be.

| EfficientNet-B0, fine-tuned, 713 held-out test photos | |
|---|---|
| Species named right | 97.2% (frozen features + logistic regression: 82.9%) |
| Macro F1 (every species counts the same) | 0.969 |
| Weakest species | Black-grass, 79.5% (confused with loose silky-bent, another grass) |
| Crops called weeds / weeds called crops | 1 of 124 / 2 of 589 |
| Right when confidence is 95% or more | 99.4% (658 photos) |
| Model | ONNX, 16 MB, CPU with onnxruntime |

- **Identify**: a sample test photo or your own upload; species, verdict, confidence and the next guesses.
- **Tray**: 72 test photos in a seed tray, each with the name given; wrong names outlined.
- **Accuracy**: the candidates, where names go wrong, recall by species, whether the confidence is honest.
- **Species**: the twelve seedlings with their recall and most common confusion.
- **Method**: data, split, training, export and limits.

## How it was built

`train.py` removes 3 duplicate photos, splits 70/15/15 by species, and compares a frozen MobileNetV3-Small with
logistic regression against MobileNetV3-Large and EfficientNet-B0 fine-tuned from torchvision's ImageNet weights (14
epochs, AdamW, one-cycle learning rate, label smoothing, rotation, flip, crop and colour augmentation). The model with
the best validation macro F1 is scored once on the test set, calibrated with one temperature on the validation photos,
and exported to ONNX; ONNX and PyTorch agree to within 1e-5 on the logits. It trains in about an hour on CPU.

The project notebook was rebuilt. The first version trained a small CNN from scratch on 64 x 64 images and reached 77%
on a 10% test set; its first model never named black-grass. The notebook re-scores the exported model on the test
photos independently.

## Why this stack

| Need | Choice |
|---|---|
| High accuracy from 3,300 training photos | **Transfer learning** from ImageNet (torchvision), fine-tuned end to end |
| A light app on a free CPU host | **ONNX** + **onnxruntime** (no PyTorch or TensorFlow at run time) |
| Honest scores | Validation chooses, the test set is scored once; macro F1 for uneven species |
| A confidence worth showing | Temperature scaling on validation photos; the app says "check by eye" below 80% |
| Trust that the shipped model is the evaluated one | **pytest** + GitHub Actions: the ONNX model reproduces the saved test predictions on the sample photos; uploads are checked; every page renders |

## Run it

```
pip install -r requirements.txt
streamlit run app.py
pip install -r requirements-dev.txt && pytest
pip install torch torchvision scikit-learn onnx
python train.py --data "<folder with images.npy and Labels.csv>"    # optional: retrain (about an hour on CPU)
```

Photos come from the Aarhus University plant seedlings dataset (T. M. Giselsson et al., 2017): lab-style top-down
shots of single seedlings. Field photos will be named less reliably. Uploaded photos are processed in memory and not
stored. SproutCheck is a portfolio demonstration.
