"""Shared control styling (buttons, segments, inputs, one focus ring), in SproutCheck's colours."""

CONTROLS = """<style>
/* Controls: one shape, one border weight and one focus ring for every button, segment and input */
:root { --k-a: #2f6b2f; --k-fill: #2f6b2f; --k-fill-2: #24531f; --k-soft: #eaf3e6; --k-ink: #1f2a1c;
        --k-line: #a9b39c; --k-panel: #ffffff; --k-r: 10px; }
[data-testid="stMain"] :is([data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-primary"],
  [data-testid="stBaseButton-secondaryFormSubmit"], [data-testid="stBaseButton-primaryFormSubmit"],
  [data-testid="stBaseButton-segmented_control"], [data-testid="stBaseButton-segmented_controlActive"],
  [data-testid="stBaseButton-pills"], [data-testid="stBaseButton-pillsActive"]) {
    border-radius: var(--k-r) !important; border: 1px solid var(--k-line) !important; background: var(--k-panel) !important;
    color: var(--k-ink) !important; box-shadow: 0 2px 0 #d6dccb !important; min-height: 2.45rem; padding: .35rem .95rem !important;
    transition: border-color .12s ease, background-color .12s ease, color .12s ease; }
[data-testid="stMain"] :is([data-testid^="stBaseButton-secondary"], [data-testid^="stBaseButton-primary"],
  [data-testid^="stBaseButton-segmented_control"], [data-testid^="stBaseButton-pills"]) p {
    font-family: 'Nunito Sans', 'Segoe UI', sans-serif; font-size: .95rem; font-weight: 600; letter-spacing: .01em; color: inherit; }
[data-testid="stMain"] :is([data-testid^="stBaseButton-secondary"], [data-testid="stBaseButton-segmented_control"],
  [data-testid="stBaseButton-pills"]):hover { border-color: var(--k-a) !important; color: var(--k-a) !important; background: var(--k-soft) !important; }
[data-testid="stMain"] :is([data-testid^="stBaseButton-primary"], [data-testid="stBaseButton-segmented_controlActive"],
  [data-testid="stBaseButton-pillsActive"]) { background: var(--k-fill) !important; border-color: var(--k-fill) !important; color: #fff !important; }
[data-testid="stMain"] :is([data-testid^="stBaseButton-primary"], [data-testid="stBaseButton-segmented_controlActive"],
  [data-testid="stBaseButton-pillsActive"]):hover { background: var(--k-fill-2) !important; border-color: var(--k-fill-2) !important; color: #fff !important; }
[data-testid="stMain"] :is([data-testid^="stBaseButton-secondary"], [data-testid^="stBaseButton-primary"]):active { transform: translate(1px, 1px); }
[data-testid="stMain"] [data-testid^="stBaseButton-"]:disabled { opacity: .45; }

/* Segmented controls: one joined strip, the chosen segment filled */
[data-testid="stMain"] [data-baseweb="button-group"] { gap: 0 !important; width: fit-content; max-width: 100%; border-radius: var(--k-r); flex-wrap: wrap; }
[data-testid="stMain"] [data-baseweb="button-group"] > button { border-radius: 0 !important; margin: 0 0 0 -1px !important; position: relative;
    padding: .35rem .8rem !important; white-space: nowrap; }
[data-testid="stMain"] [data-baseweb="button-group"] > button:first-child { margin-left: 0 !important; border-radius: var(--k-r) 0 0 var(--k-r) !important; }
[data-testid="stMain"] [data-baseweb="button-group"] > button:last-child { border-radius: 0 var(--k-r) var(--k-r) 0 !important; }
[data-testid="stMain"] [data-baseweb="button-group"] > button:hover { z-index: 1; }
[data-testid="stMain"] [data-baseweb="button-group"] > [data-testid$="Active"] { z-index: 2; }

/* The focus ring: the same 2px ring, 2px off the edge, on every control (mouse click and keyboard alike) */
[data-testid="stMain"] [data-testid^="stBaseButton-"]:focus, [data-testid="stMain"] [data-testid^="stBaseButton-"]:focus-visible,
[data-testid="stMain"] [role="slider"]:focus-visible, [data-testid="stMain"] summary:focus-visible, [data-testid="stMain"] a:focus-visible {
    outline: 2px solid var(--k-a) !important; outline-offset: 2px; }
[data-testid="stMain"] [data-baseweb="button-group"] > button:focus, [data-testid="stMain"] [data-baseweb="button-group"] > button:focus-visible { outline: none !important; }
[data-testid="stMain"] [data-baseweb="button-group"]:focus-within { outline: 2px solid var(--k-a); outline-offset: 2px; }

/* Inputs */
[data-testid="stMain"] :is([data-testid="stNumberInputContainer"], [data-testid="stTextInput"] [data-baseweb="input"],
  [data-baseweb="select"] > div, [data-baseweb="textarea"]) {
    border-radius: var(--k-r) !important; border: 1px solid var(--k-line) !important; background: var(--k-panel) !important;
    min-height: 2.45rem; transition: border-color .12s ease; }
[data-testid="stMain"] :is([data-testid="stNumberInputContainer"], [data-baseweb="select"] > div):hover { border-color: var(--k-a) !important; }
[data-testid="stMain"] :is([data-testid="stNumberInputContainer"], [data-testid="stTextInput"] [data-baseweb="input"],
  [data-baseweb="select"] > div, [data-baseweb="textarea"]):focus-within {
    border-color: var(--k-a) !important; outline: 2px solid var(--k-a); outline-offset: 2px; box-shadow: none !important; }
[data-testid="stMain"] :is([data-baseweb="input"], [data-baseweb="base-input"], input, textarea) { background: transparent !important; }
[data-testid="stMain"] [data-testid="stNumberInputContainer"] [data-baseweb="input"] { border: none !important; }
[data-testid="stMain"] textarea { font-family: 'Nunito Sans', 'Segoe UI', sans-serif; font-size: .98rem; line-height: 1.6; color: var(--k-ink); }
[data-baseweb="popover"] [data-baseweb="menu"], [data-baseweb="popover"] ul[role="listbox"] { border-radius: var(--k-r) !important; border: 1px solid var(--k-line); }
[data-baseweb="popover"] li[role="option"]:hover, [data-baseweb="popover"] li[aria-selected="true"] { background: var(--k-soft) !important; }
[data-testid="stMain"] [data-testid="stExpander"] details { border-radius: var(--k-r); border: 1px solid var(--k-line); background: var(--k-panel); }
[data-testid="stMain"] [data-testid="stExpander"] summary p { font-family: 'Nunito Sans', 'Segoe UI', sans-serif; font-weight: 700; }

/* Labels */
[data-testid="stMain"] [data-testid="stWidgetLabel"]:not([data-testid="stCheckbox"] *) p { font-size: .84rem; font-weight: 700; color: #4f5a48; }
[data-testid="stMain"] [data-testid="stAlert"] > div { border-radius: var(--k-r) !important; }
</style>"""
