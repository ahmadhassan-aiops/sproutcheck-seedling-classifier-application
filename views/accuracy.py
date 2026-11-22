"""SC-03 Accuracy: the candidates, every species' recall, the confusion matrix, crop or weed mistakes, calibration."""

from __future__ import annotations

import numpy as np
import streamlit as st

from tray import data, ui

NAMES = {"frozen_features_logreg": "Frozen MobileNetV3-Small features + logistic regression",
         "mobilenet_v3_large": "MobileNetV3-Large, fine-tuned", "efficientnet_b0": "EfficientNet-B0, fine-tuned"}


def _p(x) -> str:
    return "-" if x is None else f"{x:.0%}"


def render() -> None:
    m = data.meta()
    classes, crops = m["classes"], set(m["crops"])
    res, t = m["results"], m["test"]
    cm = np.array(m["confusion"])
    is_crop = np.array([c in crops for c in classes])
    crop_as_weed = int(cm[np.ix_(is_crop, ~is_crop)].sum())
    weed_as_crop = int(cm[np.ix_(~is_crop, is_crop)].sum())
    ui.page("SC-03", "Accuracy", "How well it names seedlings",
            f"Scored once on <b>{m['counts']['test']} test photos</b> held out from the start; the model was chosen on "
            f"{m['counts']['val']} validation photos by macro F1, which gives every species the same weight.")
    ui.tiles([("Species named right", _p(t["accuracy"]), "test photos"),
              ("Macro F1", f"{t['macro_f1']:.2f}", "every species counts the same"),
              ("Weakest species", _p(min(t["recall"].values())), min(t["recall"], key=t["recall"].get)),
              ("Crops called weeds", str(crop_as_weed), f"of {int(cm[is_crop].sum())} crop photos"),
              ("Weeds called crops", str(weed_as_crop), f"of {int(cm[~is_crop].sum())} weed photos")], highlight=0)

    with ui.panel("01", "Candidates", "leaderboard", "Validation decides; the test set is scored once, for the chosen "
                  "model and the baseline.", main=True):
        rows = ""
        for k, v in res.items():
            test = t if k == m["chosen"] else v.get("test")
            rows += (f'<tr class="{"pick" if k == m["chosen"] else ("base" if k.startswith("frozen") else "")}">'
                     f'<td>{NAMES.get(k, k)}{" (chosen)" if k == m["chosen"] else ""}</td>'
                     f'<td class="n">{_p(v["val"]["accuracy"])}</td><td class="n">{v["val"]["macro_f1"]:.3f}</td>'
                     f'<td class="n">{_p(test["accuracy"]) if test else "-"}</td>'
                     f'<td class="n">{f"{test["macro_f1"]:.3f}" if test else "-"}</td></tr>')
        st.html('<table class="tbl"><thead><tr><th>Model</th><th class="n">Val accuracy</th><th class="n">Val macro F1</th>'
                f'<th class="n">Test accuracy</th><th class="n">Test macro F1</th></tr></thead><tbody>{rows}</tbody></table>')

    with ui.panel("02", "Where names go wrong", "grid_on", "Rows: the true species; columns: the name given."):
        short = [ui.SHORT.get(c, c) for c in classes]
        body = "".join(f"<tr><td>{ui.species(a, crops)}</td>" + "".join(
            f'<td class="v {"d" if i == j else ("e" if cm[i, j] else "")}">{int(cm[i, j]) or ""}</td>' for j in range(len(classes)))
            + "</tr>" for i, a in enumerate(classes))
        st.html('<div style="overflow-x:auto"><table class="tbl mx"><thead><tr><th>True</th>'
                + "".join(f'<th class="c" title="{ui.esc(c)}">{ui.esc(s)}</th>' for c, s in zip(classes, short))
                + f"</tr></thead><tbody>{body}</tbody></table></div>")
        off = [(cm[i, j], classes[i], classes[j]) for i in range(len(classes)) for j in range(len(classes)) if i != j and cm[i, j]]
        off.sort(reverse=True)
        if off:
            ui.notes(*[f"{n} {ui.esc(a)} named {ui.esc(b)}" for n, a, b in off[:4]])
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        with ui.panel("03", "Recall by species", "eco", "Share of each species' test photos named correctly."):
            rows = "".join(f'<tr><td>{ui.species(c, crops)}</td><td class="n">{_p(t["recall"][c])}</td></tr>'
                           for c in sorted(classes, key=lambda c: t["recall"][c]))
            st.html(f'<table class="tbl"><thead><tr><th>Species</th><th class="n">Recall</th></tr></thead><tbody>{rows}</tbody></table>')
    with c2:
        with ui.panel("04", "Is the confidence honest?", "speed", "Test photos grouped by the confidence shown."):
            rows = "".join(f'<tr><td>{b["band"]}</td><td class="n">{b["n"]}</td><td class="n">{_p(b["mean_confidence"])}</td>'
                           f'<td class="n">{_p(b["accuracy"])}</td></tr>' for b in m["calibration"])
            st.html('<table class="tbl"><thead><tr><th>Confidence</th><th class="n">Photos</th><th class="n">Average shown</th>'
                    f'<th class="n">Named right</th></tr></thead><tbody>{rows}</tbody></table>')
    ui.footer()
