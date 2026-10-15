from pathlib import Path
import joblib

import streamlit as st
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt

# ========================================
# Load Cat & Dog AI Model
# ========================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "cat_dog_model.joblib"

model = joblib.load(MODEL_PATH)


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
:root {
  --audio-bg: #FFF8F2;
  --audio-surface: #FFFCF9;
  --audio-tab: #F3E5DB;
  --audio-tab-hover: #EED5C8;
  --audio-active: #FFE1D4;
  --audio-accent: #D65D50;
  --audio-text: #633D32;
  --audio-muted: #89695D;
  --audio-border: #E7CFC2;
  --audio-title: #633D32;
  --audio-kicker: #C9654D;
}

html, body, .stApp, [data-testid="stAppViewContainer"] {
  background: var(--audio-bg) !important;
}

/* Four-tab navigation: warm cards, strong contrast, and a clear active state. */
div[data-testid="stTabs"] { margin-top: 14px !important; }
div[data-testid="stTabs"] [data-baseweb="tab-list"],
div[data-testid="stTabs"] [role="tablist"] {
  display: flex !important;
  flex-wrap: wrap !important;
  align-items: stretch !important;
  gap: 10px !important;
  padding: 10px 2px 12px !important;
  background: transparent !important;
  border-bottom: 1px solid var(--audio-border) !important;
}

div[data-testid="stTabs"] button[data-baseweb="tab"],
div[data-testid="stTabs"] [role="tab"] {
  appearance: none !important;
  box-sizing: border-box !important;
  min-height: 52px !important;
  padding: 11px 18px !important;
  margin: 0 !important;
  border: 1px solid var(--audio-border) !important;
  border-bottom: 3px solid transparent !important;
  border-radius: 12px !important;
  background: var(--audio-tab) !important;
  color: var(--audio-text) !important;
  -webkit-text-fill-color: var(--audio-text) !important;
  font-size: 16px !important;
  font-weight: 750 !important;
  line-height: 1.35 !important;
  opacity: 1 !important;
  box-shadow: 0 2px 5px rgba(99, 61, 50, .05) !important;
  transition: background-color .15s ease, border-color .15s ease, transform .15s ease !important;
}

/* Streamlit versions wrap tab labels in different nested elements; force readable color on all. */
div[data-testid="stTabs"] button[data-baseweb="tab"] *,
div[data-testid="stTabs"] [role="tab"] *,
div[data-testid="stTabs"] button[data-baseweb="tab"] p,
div[data-testid="stTabs"] [role="tab"] p,
div[data-testid="stTabs"] button[data-baseweb="tab"] span,
div[data-testid="stTabs"] [role="tab"] span,
div[data-testid="stTabs"] button[data-baseweb="tab"] svg,
div[data-testid="stTabs"] [role="tab"] svg {
  color: var(--audio-text) !important;
  -webkit-text-fill-color: var(--audio-text) !important;
  fill: var(--audio-text) !important;
  opacity: 1 !important;
  text-shadow: none !important;
  font-weight: 750 !important;
}

div[data-testid="stTabs"] button[data-baseweb="tab"]:hover,
div[data-testid="stTabs"] [role="tab"]:hover {
  background: var(--audio-tab-hover) !important;
  border-color: #D7AD9D !important;
  transform: translateY(-1px) !important;
}

div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"],
div[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
