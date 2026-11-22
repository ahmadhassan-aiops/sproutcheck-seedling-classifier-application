"""SC-02 Tray: test photos laid out in a seed tray, each with the species the model named."""

from __future__ import annotations

import streamlit as st

from tray import data, ui


def render() -> None:
    m = data.meta()
    crops = set(m["crops"])
    pred = data.predictions().set_index("image_id")
    s = data.samples()
    s = s.join(pred[["pred", "confidence", "right"]], on="image_id")
    ui.page("SC-02", "Tray", "A tray of test seedlings",
            f"{len(s)} of the {m['counts']['test']} test photos, six per species, none used in training. Each cell shows "
            "the species the model named; a red edge marks a wrong name.")
    show = st.segmented_control("Show", ["All", "Crops", "Weeds", "Wrong only"], default="All", key="t_show")
    view = s
    if show == "Crops":
        view = s[s["label"].isin(crops)]
    elif show == "Weeds":
        view = s[~s["label"].isin(crops)]
    elif show == "Wrong only":
        view = s[~s["right"]]
    with ui.panel("01", "The tray", "grid_view", "Hover a cell for the true species and the confidence.", main=True):
        cells = ""
        for r in view.itertuples():
            c = ui.CROP if r.pred in crops else ui.WEED
            cells += (f'<div class="cell{"" if r.right else " x"}" title="True: {ui.esc(r.label)} | named: {ui.esc(r.pred)} '
                      f'({r.confidence:.0%})"><img src="{data.data_uri(data.sample_image(r.file), 160)}" alt="{ui.esc(r.label)}">'
                      f'<span class="tag" style="--c:{c}">{ui.esc(ui.SHORT.get(r.pred, r.pred))}</span></div>')
        st.html(f'<div class="trayg">{cells}</div>' if cells else '<p class="sub">No photos to show.</p>')
        wrong = s[~s["right"]]
        ui.takeaway(f"{int(s['right'].sum())} of {len(s)} named correctly in this tray. "
                    + ("Wrong names: " + "; ".join(f"{ui.esc(r.label)} named {ui.esc(r.pred)}" for r in wrong.itertuples())
                       if len(wrong) else "No wrong names in this tray."))
    ui.footer()
