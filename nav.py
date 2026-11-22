"""Page registry (plant-label row, native navigation hidden)."""

import streamlit as st

from views import accuracy, identify, method, tray

IDENTIFY = st.Page(identify.render, title="Identify", icon=":material/eco:", url_path="identify", default=True)
TRAY = st.Page(tray.render, title="Tray", icon=":material/grid_view:", url_path="tray")
ACCURACY = st.Page(accuracy.render, title="Accuracy", icon=":material/leaderboard:", url_path="accuracy")
SPECIES = st.Page(method.species_page, title="Species", icon=":material/local_florist:", url_path="species")
METHOD = st.Page(method.render, title="Method", icon=":material/description:", url_path="method")

PAGES = [IDENTIFY, TRAY, ACCURACY, SPECIES, METHOD]
