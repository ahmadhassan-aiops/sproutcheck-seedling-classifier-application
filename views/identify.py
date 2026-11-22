"""SC-01 Identify: one seedling photo, its species, crop or weed, and the look-alikes the model also considered."""

from __future__ import annotations

import streamlit as st

from sprout.model import open_image
from tray import data, ui


def render() -> None:
    m = data.meta()
    clf = data.classifier()
    crops = clf.crops
    t = m["test"]
    ui.page("SC-01", "Identify", "Crop or weed?",
            f"Pick a seedling photo or upload your own. A fine-tuned <b>{ui.MODEL_NAME.get(m['chosen'], m['chosen'])}</b> names the "
            f"species among 12, says whether it is a crop to keep or a weed to remove, and shows how sure it is. On "
            f"{m['counts']['test']} photos it never saw in training it named {t['accuracy']:.0%} correctly.")
    src = st.segmented_control("Photo from", ["Sample tray", "Upload"], default="Sample tray", key="i_src")
    image, label = None, None
    if src == "Upload":
        up = st.file_uploader("Seedling photo (JPG or PNG, taken from above)", type=["jpg", "jpeg", "png", "webp"], key="i_up")
        ui.caution("The model learned from top-down photos of single seedlings on dark soil in trays. Photos in a field, "
                   "from the side or with several plants will be named less reliably.", "photo_camera")
        if up is not None:
            try:
                image = open_image(up.getvalue())
            except ValueError as err:
                st.error(str(err), icon=":material/broken_image:")
    else:
        s = data.samples()
        sp = st.selectbox("Species in the sample tray", m["classes"], key="i_species")
        files = list(s.loc[s["label"] == sp, "file"])
        pick = st.radio("Photo", list(range(len(files))), horizontal=True, key="i_pick", format_func=lambda i: f"#{i + 1}")
        image, label = data.sample_image(files[pick or 0]), sp
    if image is None:
        st.info("Upload a photo to identify it.", icon=":material/upload:")
        ui.footer()
        return

    probs = clf.probabilities(image)
    top, p = next(iter(probs.items()))
    left, right = st.columns([1, 1.25], gap="medium")
    with left, ui.panel("01", "The seedling", "photo_camera"):
        st.html(f'<div class="photo"><img src="{data.data_uri(image, 420)}" alt="seedling photo"></div>')
        if label:
            ok = label == top
            st.html(f'<p class="sub" style="margin:.6rem 0 0">Test photo (never used in training). True species: '
                    f'{ui.species(label, crops)} <b style="color:{"#2f6b2f" if ok else "#a4471f"}">'
                    f'{"named correctly" if ok else "named wrongly"}</b></p>')
    with right, ui.panel("02", "What it is", "eco", main=True):
        band = "sure" if p >= .95 else "fairly sure" if p >= .8 else "unsure: check by eye"
        st.html(f'<p style="margin:0 0 .5rem">{ui.verdict(top, crops)}</p>'
                f'<p style="font-family:Gloock,Georgia,serif;font-size:1.7rem;margin:.2rem 0">{ui.esc(top)}</p>'
                f'<p class="sub" style="margin:0 0 .9rem">{p:.0%} confidence, {band}.</p>')
        rows = "".join(f'<span>{ui.species(c, crops)}</span><span class="bar"><i style="width:{q * 100:.1f}%;--c:'
                       f'{ui.CROP if c in crops else ui.WEED}"></i></span><span class="v">{q:.0%}</span>'
                       for c, q in list(probs.items())[:4])
        st.html(f'<div class="guess">{rows}</div>')
        second, q2 = list(probs.items())[1]
        if q2 >= .1:
            flip = (second in crops) != (top in crops)
            ui.caution(f"Also possible: <b>{ui.esc(second)}</b> ({q2:.0%})."
                       + (" That would change the verdict, so look before pulling it." if flip else ""), "visibility")
    ui.footer()
