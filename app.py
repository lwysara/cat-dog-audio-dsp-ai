from pathlib import Path
import io
import numpy as np
import pandas as pd
import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import joblib
from scipy.signal import butter, sosfilt

st.set_page_config(page_title="Cat & Dog Audio DSP AI", page_icon="🐱", layout="wide")

# ------------------------------------------------------------------
# Warm, high-contrast UI theme for the four Audio DSP tabs
# ------------------------------------------------------------------
st.markdown("""
<style>
/* Warm palette: use strong selectors because Streamlit/BaseWeb themes can override tabs. */
:root {
  --audio-bg: #FFF8F2;
  --audio-tab-bg: #F8EDE5;
  --audio-active-bg: #FFE5D9;
  --audio-accent: #D95F50;
  --audio-text: #633D32;
  --audio-border: #E8CFC1;
}
html, body, .stApp, [data-testid="stAppViewContainer"] {
  background: var(--audio-bg) !important;
}

/* Tab bar background and spacing */
.stTabs, [data-testid="stTabs"] { margin-top: 12px; }
.stTabs [data-baseweb="tab-list"],
[data-testid="stTabs"] [role="tablist"] {
  display: flex !important;
  flex-wrap: wrap !important;
  align-items: stretch !important;
  gap: 8px !important;
  background: var(--audio-bg) !important;
  padding: 8px 0 0 !important;
  border-bottom: 2px solid var(--audio-border) !important;
}

/* Target both current and alternate Streamlit tab markup */
.stTabs button[data-baseweb="tab"],
.stTabs [role="tab"],
[data-testid="stTabs"] button[role="tab"] {
  box-sizing: border-box !important;
  min-height: 54px !important;
  padding: 12px 18px !important;
  margin: 0 !important;
  background: var(--audio-tab-bg) !important;
  color: var(--audio-text) !important;
  border: 1px solid var(--audio-border) !important;
  border-bottom: 4px solid transparent !important;
  border-radius: 10px 10px 0 0 !important;
  font-size: 17px !important;
  font-weight: 750 !important;
  line-height: 1.45 !important;
  opacity: 1 !important;
  box-shadow: none !important;
}
.stTabs button[data-baseweb="tab"] *,
.stTabs [role="tab"] *,
[data-testid="stTabs"] button[role="tab"] * {
  color: var(--audio-text) !important;
  font-size: 17px !important;
  font-weight: 750 !important;
  opacity: 1 !important;
  fill: var(--audio-text) !important;
}
.stTabs button[data-baseweb="tab"]:hover,
.stTabs [role="tab"]:hover,
[data-testid="stTabs"] button[role="tab"]:hover {
  background: #F4DCCF !important;
  border-color: #D9AFA0 !important;
}
.stTabs button[data-baseweb="tab"][aria-selected="true"],
.stTabs [role="tab"][aria-selected="true"],
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
  background: var(--audio-active-bg) !important;
  color: #8E382D !important;
  border: 1px solid #E8B5A5 !important;
  border-bottom: 4px solid var(--audio-accent) !important;
  font-weight: 800 !important;
}
.stTabs button[data-baseweb="tab"][aria-selected="true"] *,
.stTabs [role="tab"][aria-selected="true"] *,
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] * {
  color: #8E382D !important;
  font-weight: 800 !important;
  opacity: 1 !important;
  fill: #8E382D !important;
}
.stTabs button[data-baseweb="tab"]:focus-visible,
