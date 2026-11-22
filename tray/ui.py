"""SproutCheck components: a greenhouse bench. Navigation is a row of plant labels; every block is a white tray card on
a soil-brown bench edge; species names are written on stake labels; the crop or weed verdict is a coloured tag."""

from __future__ import annotations

import html
from contextlib import contextmanager

import streamlit as st

from tray._controls import CONTROLS

CODE_URL = "https://github.com/ahmadhassan-aiops/sproutcheck-seedling-classifier-application"
NOTEBOOK_URL = ("https://github.com/ahmadhassan-aiops/AIOPS-Portfolio-Projects/blob/main/4.%20Computer%20Vision%20Projects/"
                "5.%20CV%20Project/Seedling_Classification.ipynb")
LEARN_URL = "https://learn.ahmadhassanrx.com"
CREDIT = ("Images: 4,750 photos of 12 plant species at the seedling stage, 128 x 128 pixels, from the Aarhus University "
          "plant seedlings dataset (T. M. Giselsson et al., 2017). Models: ImageNet weights from torchvision, fine-tuned.")

CROP, WEED = "#2f6b2f", "#a4471f"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Gloock&family=Nunito+Sans:opsz,wght@6..12,400;6..12,600;6..12,700;6..12,800&family=Kalam:wght@400;700&display=swap');
:root { --bench:#efeadf; --card:#ffffff; --ink:#1f2a1c; --muted:#4f5a48; --faint:#6e7866; --line:#d6dccb; --soil:#6b4f3a;
        --leaf:#2f6b2f; --leaf-soft:#eaf3e6; --weed:#a4471f; --weed-soft:#fbece5; }
html, body, [data-testid="stAppViewContainer"] { font-family: 'Nunito Sans', 'Segoe UI', sans-serif; color: var(--ink); }
[data-testid="stAppViewContainer"], [data-testid="stMain"] { background: var(--bench); }
.block-container { padding-top: 4.1rem; max-width: 1240px; }
header[data-testid="stHeader"] { background: var(--bench); }
.msr { font-family: 'Material Symbols Rounded'; font-weight: normal; font-style: normal; line-height: 1; display: inline-block;
       -webkit-font-feature-settings: 'liga'; font-feature-settings: 'liga'; -webkit-font-smoothing: antialiased; }

/* Tray cards on a soil edge */
[data-testid="stVerticalBlock"][data-test-scroll-behavior] { border: 1px solid var(--line) !important; border-bottom: 6px solid var(--soil) !important;
    border-radius: 14px !important; background: var(--card); box-shadow: 0 2px 6px rgba(31, 42, 28, .08); padding: 1.1rem 1.25rem 1.1rem !important; }
[data-testid="stVerticalBlock"][data-test-scroll-behavior][class*="st-key-main_"] { border-bottom-color: var(--leaf) !important;
    box-shadow: 0 0 0 2px #cfe3c8, 0 6px 16px -8px rgba(47, 107, 47, .45) !important; }

/* Navigation: plant labels */
[data-testid="stHorizontalBlock"]:has(.brand) { gap: .45rem; align-items: center; margin-bottom: .2rem; padding-bottom: .7rem;
    border-bottom: 6px solid var(--soil); border-radius: 0 0 3px 3px; }
.brand { display: flex; align-items: center; gap: .6rem; }
.brand .mk { width: 2.6rem; height: 2.6rem; border-radius: 50% 50% 50% 8px; background: var(--leaf); color: #fff; display: grid; place-items: center; }
.brand .mk .msr { font-size: 1.5rem; }
.brand .name { font-family: 'Gloock', Georgia, serif; font-size: 1.7rem; margin: 0; line-height: 1; }
.brand .tagline { font-size: .78rem; color: var(--muted); margin: .2rem 0 0; }
[data-testid="stHorizontalBlock"]:has(.brand) [data-testid="stPageLink"] a { justify-content: center; background: #fff; border: 1px solid var(--line);
    border-radius: 8px 8px 22px 22px; padding: .45rem .5rem .55rem; box-shadow: 0 2px 0 var(--line); }
[data-testid="stHorizontalBlock"]:has(.brand) [data-testid="stPageLink"] a p { font-family: 'Kalam', cursive; font-weight: 700; font-size: 1.08rem;
    color: var(--ink); white-space: nowrap; }
[data-testid="stHorizontalBlock"]:has(.brand) [data-testid="stPageLink"] a span { color: var(--leaf); }
[data-testid="stHorizontalBlock"]:has(.brand) [data-testid="stPageLink"] a:hover { border-color: var(--leaf); }
[data-testid="stHorizontalBlock"]:has(.brand) [data-testid="stPageLink"] a:focus-visible { outline: 2px solid var(--leaf) !important; outline-offset: 2px; }

.kicker { font-size: .8rem; color: var(--muted); margin: 1.3rem 0 .35rem; font-weight: 700; letter-spacing: .04em; }
.kicker b { color: var(--leaf); }
.title { font-family: 'Gloock', Georgia, serif; font-size: 2.5rem; font-weight: 400; margin: 0; line-height: 1.1; }
.lead { color: var(--muted); font-size: 1.05rem; line-height: 1.6; max-width: 820px; margin: .55rem 0 1.1rem; }
.lead b { color: var(--ink); }

/* Card title: a stake label */
.stake { display: flex; align-items: center; gap: .6rem; margin: 0 0 .75rem; }
.stake .lab { display: inline-flex; align-items: center; gap: .45rem; background: #fff; border: 1px solid var(--line); border-left: 4px solid var(--leaf);
              border-radius: 3px 14px 14px 3px; padding: .2rem .8rem .2rem .55rem; }
.stake .no { font-size: .72rem; font-weight: 800; color: var(--leaf); }
.stake .t { font-family: 'Kalam', cursive; font-weight: 700; font-size: 1.22rem; margin: 0; line-height: 1.2; }
.stake .ic { margin-left: auto; color: var(--faint); }
.stake .ic .msr { font-size: 1.35rem; }
.sub { font-size: .92rem; color: var(--muted); margin: -.2rem 0 .85rem; line-height: 1.5; }

.tiles { display: grid; grid-template-columns: repeat(var(--n), minmax(0, 1fr)); gap: .6rem; margin: 0 0 1rem; }
@media (max-width: 860px) { .tiles { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
.tile { background: #fff; border: 1px solid var(--line); border-radius: 12px; padding: .6rem .85rem .65rem; }
.tile.hl { border-color: var(--leaf); box-shadow: inset 0 -4px 0 var(--leaf); }
.tile-l { font-size: .8rem; color: var(--muted); margin: 0; font-weight: 700; }
.tile-v { font-family: 'Gloock', Georgia, serif; font-size: 1.95rem; margin: .1rem 0; line-height: 1.1; }
.tile-n { font-size: .8rem; color: var(--faint); margin: 0; line-height: 1.4; }

/* Verdict tag and species stakes */
.verdict { display: inline-flex; align-items: center; gap: .4rem; color: #fff; background: var(--c); border-radius: 999px; padding: .35rem 1rem .35rem .7rem;
           font-weight: 800; font-size: 1.05rem; }
.verdict .msr { font-size: 1.3rem; }
.sp { display: inline-flex; align-items: center; gap: .3rem; font-weight: 700; font-size: .86rem; border: 1px solid var(--c); color: var(--c);
      background: var(--s); border-radius: 3px 12px 12px 3px; padding: .06rem .55rem .06rem .45rem; white-space: nowrap; }
.sp .msr { font-size: .95rem; }
.guess { display: grid; grid-template-columns: minmax(9rem, 38%) 1fr 3.4rem; gap: .35rem .6rem; align-items: center; font-size: .92rem; }
.guess .bar { height: 12px; background: #eef1e8; border-radius: 6px; overflow: hidden; }
.guess .bar i { display: block; height: 100%; background: var(--c); border-radius: 6px; }
.guess .v { font-weight: 800; text-align: right; }
.photo img { width: 100%; max-width: 340px; display: block; margin: 0 auto; border-radius: 10px; border: 1px solid var(--line); image-rendering: auto; }

/* Seed tray grid */
.trayg { display: grid; grid-template-columns: repeat(auto-fill, minmax(92px, 1fr)); gap: 6px; background: #3c2d22; padding: 8px; border-radius: 10px; }
.cell { background: #2b2019; border-radius: 6px; padding: 4px; position: relative; }
.cell img { width: 100%; display: block; border-radius: 4px; }
.cell .tag { position: absolute; left: 6px; bottom: 6px; font-size: .68rem; font-weight: 800; color: #fff; background: var(--c); border-radius: 4px;
             padding: 0 .3rem; }
.cell.x { box-shadow: 0 0 0 3px var(--weed); }

.tbl { width: 100%; border-collapse: separate; border-spacing: 0; font-size: .88rem; border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }
.tbl th { font-size: .74rem; font-weight: 800; letter-spacing: .03em; text-transform: uppercase; color: #fff; text-align: left; padding: .5rem .6rem; background: var(--soil); }
.tbl td { padding: .45rem .6rem; border-top: 1px solid #edf0e6; vertical-align: top; }
.tbl td:first-child { white-space: nowrap; }
.tbl th.n, .tbl td.n { text-align: right; }
.tbl td.n { font-weight: 700; font-variant-numeric: tabular-nums; }
.tbl tbody tr:nth-child(even) td { background: #f8faf4; }
.tbl tbody tr.pick td { background: var(--leaf-soft); font-weight: 800; }
.tbl tbody tr.base td { color: var(--muted); font-style: italic; }
.mx td.v { text-align: center; font-weight: 700; font-variant-numeric: tabular-nums; padding: .35rem .2rem; }
.mx td.v.d { background: var(--leaf-soft) !important; color: var(--leaf); }
.mx td.v.e { background: var(--weed-soft) !important; color: var(--weed); }
.mx th.c { text-align: center; padding: .4rem .2rem; font-size: .66rem; }

.notes { margin: .9rem 0 0; border: 1px dashed #b9c4aa; background: #fbfdf8; padding: .55rem .9rem .4rem; border-radius: 10px; }
.notes .h { font-family: 'Kalam', cursive; font-size: 1rem; font-weight: 700; color: var(--leaf); margin: 0 0 .2rem; display: flex; gap: .35rem; align-items: center; }
.notes ul { margin: 0; padding-left: 1.1rem; }
.notes li { font-size: .92rem; line-height: 1.55; margin: 0 0 .35rem; }
.takeaway { display: flex; gap: .7rem; align-items: flex-start; margin: .9rem 0 0; padding: .7rem .9rem; background: var(--leaf-soft);
            border-radius: 10px; font-size: .98rem; line-height: 1.55; }
.takeaway .msr { color: var(--leaf); font-size: 1.4rem; }
.caution { display: flex; gap: .55rem; border: 1px solid #ecd9a6; padding: .6rem .8rem; margin: .8rem 0 0; font-size: .9rem; border-radius: 10px;
           line-height: 1.5; background: #fffaf0; }
.caution .msr { color: #8a5a00; font-size: 1.2rem; }
.facts { display: grid; grid-template-columns: minmax(9rem, 28%) 1fr; margin: 0; padding: 0 !important; }
.facts dt, .facts dd { padding: .5rem .3rem; border-bottom: 1px solid #edf0e6; margin: 0; font-size: .92rem; line-height: 1.5; }
.facts dt { font-weight: 800; color: var(--muted); font-size: .84rem; }
.foot { margin: 2rem 0 .5rem; border-top: 6px solid var(--soil); display: flex; flex-wrap: wrap; font-size: .86rem; color: var(--muted); }
.foot > * { padding: .6rem 1rem .6rem 0; margin-right: 1rem; }
.foot b { color: var(--ink); }
.foot a { color: var(--leaf); font-weight: 800; text-decoration: none; }
.foot a:hover { text-decoration: underline; }
.foot .credit { flex-basis: 100%; padding-top: 0; font-size: .8rem; color: var(--faint); }
</style>
"""


SHORT = {"Black-grass": "Black-grass", "Charlock": "Charlock", "Cleavers": "Cleavers", "Common Chickweed": "Chickweed",
         "Common wheat": "Wheat", "Fat Hen": "Fat hen", "Loose Silky-bent": "Silky-bent", "Maize": "Maize",
         "Scentless Mayweed": "Mayweed", "Shepherds Purse": "Shep. purse", "Small-flowered Cranesbill": "Cranesbill",
         "Sugar beet": "Sugar beet"}
MODEL_NAME = {"efficientnet_b0": "EfficientNet-B0", "mobilenet_v3_large": "MobileNetV3-Large"}


def inject_css() -> None:
    st.html(CSS + CONTROLS)


def icon(name: str) -> str:
    return f'<span class="msr" aria-hidden="true">{name}</span>'


def esc(text) -> str:
    return html.escape(str(text))


def page(code: str, name: str, title: str, lead: str) -> None:
    st.html(f'<p class="kicker"><b>{esc(code)}</b> · {esc(name)}</p><h1 class="title">{esc(title)}</h1><p class="lead">{lead}</p>')


@contextmanager
def panel(no: str, title: str, pictogram: str, sub: str = "", main: bool = False):
    with st.container(border=True, key=f"main_{no}" if main else None):
        st.html(f'<div class="stake"><span class="lab"><span class="no">{esc(no)}</span><p class="t">{esc(title)}</p></span>'
                f'<span class="ic">{icon(pictogram)}</span></div>' + (f'<p class="sub">{sub}</p>' if sub else ""))
        yield


def tiles(items, highlight=None) -> None:
    body = "".join(f'<div class="tile{" hl" if i == highlight else ""}"><p class="tile-l">{esc(l)}</p>'
                   f'<p class="tile-v">{v}</p><p class="tile-n">{n}</p></div>' for i, (l, v, n) in enumerate(items))
    st.html(f'<div class="tiles" style="--n:{len(items)}">{body}</div>')


def is_crop(species: str, crops) -> bool:
    return species in crops


def species(name: str, crops) -> str:
    c = CROP if name in crops else WEED
    return (f'<span class="sp" style="--c:{c};--s:{"#eaf3e6" if name in crops else "#fbece5"}">'
            f'{icon("agriculture" if name in crops else "grass")}{esc(name)}</span>')


def verdict(name: str, crops) -> str:
    crop = name in crops
    return (f'<span class="verdict" style="--c:{CROP if crop else WEED}">{icon("agriculture" if crop else "grass")}'
            f'{"Crop: keep it" if crop else "Weed: remove it"}</span>')


def notes(*items: str) -> None:
    st.html(f'<div class="notes"><p class="h">{icon("edit_note")}Bench notes</p><ul>' + "".join(f"<li>{i}</li>" for i in items) + "</ul></div>")


def takeaway(text: str) -> None:
    st.html(f'<div class="takeaway">{icon("lightbulb")}<div>{text}</div></div>')


def caution(text: str, pictogram: str = "info") -> None:
    st.html(f'<div class="caution">{icon(pictogram)}<div>{text}</div></div>')


def footer() -> None:
    st.html(f'<div class="foot"><span>SproutCheck · <b>Ahmad Hassan</b>, AI Engineer</span>'
            f'<a href="{NOTEBOOK_URL}" target="_blank">Notebook</a><a href="{CODE_URL}" target="_blank">App code</a>'
            f'<a href="{LEARN_URL}" target="_blank">Learn with Ahmad</a><span class="credit">{esc(CREDIT)}</span></div>')
