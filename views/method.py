"""SC-04 Species and SC-05 Method."""

from __future__ import annotations

import streamlit as st

from tray import data, ui


def species_page() -> None:
    m = data.meta()
    crops = set(m["crops"])
    cm = m["confusion"]
    classes = m["classes"]
    s = data.samples().join(data.predictions().set_index("image_id")[["confidence", "right"]], on="image_id")
    best = s[s["right"]].sort_values("confidence").groupby("label")["file"].last().to_dict()   # clearest example
    ui.page("SC-04", "Species", "The twelve seedlings",
            "Three crops to keep and nine weeds to remove, with how often the model names each and what it confuses "
            "each one with.")
    for group, members in [("Crops", [c for c in classes if c in crops]), ("Weeds", [c for c in classes if c not in crops])]:
        with ui.panel("01" if group == "Crops" else "02", group, "agriculture" if group == "Crops" else "grass",
                      main=group == "Crops"):
            cols = st.columns(3)
            for k, c in enumerate(members):
                i = classes.index(c)
                row = cm[i]
                wrong = sorted(((n, classes[j]) for j, n in enumerate(row) if j != i and n), reverse=True)
                f = best.get(c) or s.loc[s["label"] == c, "file"].iloc[0]
                conf = f"Most often confused with {ui.esc(wrong[0][1])} ({wrong[0][0]} of {sum(row)})." if wrong else "No confusions on the test photos."
                cols[k % 3].html(f'<div class="photo"><img src="{data.data_uri(data.sample_image(f), 220)}" alt="{ui.esc(c)}"></div>'
                                 f'<p style="margin:.45rem 0 .2rem">{ui.species(c, crops)}</p><p class="sub" style="margin:0 0 .9rem">'
                                 f'{m["counts"]["per_class"][c]} photos · recall {m["test"]["recall"][c]:.0%}. {conf}</p>')
    ui.footer()


def render() -> None:
    m = data.meta()
    c = m["counts"]
    chosen = m["results"][m["chosen"]]
    facts = [
        ("Data", f"{c['images'] + c['duplicates_removed']:,} photos of 12 species at the seedling stage, 128 x 128 pixels, "
                 f"top-down on soil in trays (Aarhus University plant seedlings dataset). {c['duplicates_removed']} exact "
                 "duplicate photos were kept once, so none can sit in both training and test."),
        ("Split", f"Stratified by species: {c['train']:,} training, {c['val']} validation, {c['test']} test photos. The "
                  "test photos were scored once, after the model was chosen."),
        ("Candidates", "Frozen MobileNetV3-Small features with logistic regression (baseline), and MobileNetV3-Large and "
                       f"EfficientNet-B0 fine-tuned from ImageNet weights for {m['epochs']} epochs (best epoch by validation "
                       "macro F1). Training images were randomly rotated, flipped, cropped and colour-shifted; seedlings "
                       "photographed from above have no up or down."),
        ("Chosen", f"{ui.MODEL_NAME.get(m['chosen'], m['chosen'])}, {chosen['parameters'] / 1e6:.1f} million parameters, exported to "
                   f"ONNX ({m['onnx_mb']} MB) and run on CPU with onnxruntime; ONNX and PyTorch agree to "
                   f"{m['onnx_max_diff']:.0e} on the logits."),
        ("Confidence", f"Calibrated on the validation photos with one temperature ({m['temperature']:.2f})."),
        ("Crop or weed", "Maize, common wheat and sugar beet are crops; the other nine are weeds."),
        ("Limits", "Lab-style photos: field photos, side views or several plants in one photo will score lower. Twelve "
                   "species only; any other plant is still given one of these twelve names. Uploaded photos are "
                   "processed in memory and not stored."),
    ]
    ui.page("SC-05", "Method", "Data and method", "What SproutCheck learned from, how it was tested, and what it cannot do.")
    with ui.panel("01", "Facts", "description", main=True):
        st.html('<dl class="facts">' + "".join(f"<dt>{ui.esc(k)}</dt><dd>{v}</dd>" for k, v in facts) + "</dl>")
    ui.footer()
