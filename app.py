"""
SproutCheck: names a seedling from a photo (12 species), says whether it is a crop or a weed, and shows how sure it is.
A fine-tuned ImageNet model exported to ONNX and run on CPU; trained by train.py, inference code in sprout/.
"""

import streamlit as st

st.set_page_config(page_title="SproutCheck | Seedling identification | Ahmad Hassan", page_icon=":material/eco:",
                   layout="wide", initial_sidebar_state="collapsed")

from nav import PAGES  # noqa: E402
from tray.ui import icon, inject_css  # noqa: E402

inject_css()
current = st.navigation(PAGES, position="hidden")
cols = st.columns([1.9, 1, 1, 1, 1, 1], vertical_alignment="center")
cols[0].html(f'<div class="brand"><span class="mk">{icon("eco")}</span><div><p class="name">SproutCheck</p>'
             '<p class="tagline">Seedling photos: crop or weed</p></div></div>')
for col, p in zip(cols[1:], PAGES):
    col.page_link(p, label=p.title, icon=p.icon, width="stretch")
k = [p.title for p in PAGES].index(current.title) + 2
st.html(f'<style>[data-testid="stHorizontalBlock"]:has(.brand) > [data-testid="stColumn"]:nth-child({k}) a '
        '{ background: #2f6b2f !important; border-color: #24531f !important; } '
        f'[data-testid="stHorizontalBlock"]:has(.brand) > [data-testid="stColumn"]:nth-child({k}) a p, '
        f'[data-testid="stHorizontalBlock"]:has(.brand) > [data-testid="stColumn"]:nth-child({k}) a span '
        '{ color: #fff !important; }</style>')
current.run()
