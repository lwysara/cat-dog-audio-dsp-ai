from pathlib import Path
import hashlib
import io

import joblib
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.signal import butter, sosfilt

# ============================================================
# Audio DSP AI ＋ Machine Learning — Phase 1 integration
# Keeps the existing model bundle contract and DSP explanations.
# ============================================================

st.set_page_config(
    page_title="Audio DSP AI ＋ Machine Learning",
    page_icon="🎙️",
    layout="wide",
)

# -------------------- Modern GUI theme --------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Noto+Sans+TC:wght@400;500;600;700;800&display=swap');
:root {
  --ink: #633D32;
  --muted: #8A6A5D;
  --accent: #C9654D;
  --accent-dark: #8E382D;
  --line: #EED5C6;
  --page: #FFF8F2;
  --surface: #FFFCF9;
  --tab-bg: #F8EDE5;
  --tab-active: #FFE5D9;
}
html, body, [class*="css"] { font-family: Inter, "Noto Sans TC", sans-serif; }
html, body, .stApp, [data-testid="stAppViewContainer"] {
  background: var(--page) !important;
  color: var(--ink) !important;
}
.block-container { max-width: 1500px; padding-top: 1.5rem; padding-bottom: 3rem; }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#8E5140 0%,#633D32 100%) !important; border-right: 1px solid #D9B5A3; }
[data-testid="stSidebar"] * { color: #FFF8F2; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #F7E4D9; }
h1, h2, h3 { color: var(--ink) !important; letter-spacing: -.025em; }
h1 { font-weight: 800 !important; }
[data-testid="stMetric"] { background: linear-gradient(135deg,#FFFCF9,#F8EDE5) !important; border: 1px solid var(--line); padding: 18px 20px; border-radius: 18px; box-shadow: 0 8px 24px rgba(99,61,50,.06); }
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-weight: 600; }
[data-testid="stMetricValue"] { color: var(--accent-dark) !important; font-weight: 800; }
div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 18px; }
.stButton > button[kind="primary"] { background: linear-gradient(90deg,#D97860,#E85D4F) !important; color: #FFFFFF !important; border: 0; border-radius: 12px; min-height: 3rem; font-weight: 700; box-shadow: 0 7px 18px rgba(217,95,80,.18); }
.stButton > button[kind="primary"]:hover { border: 0; filter: brightness(1.04); box-shadow: 0 9px 24px rgba(217,95,80,.24); }
.stButton > button, .stDownloadButton > button { border-radius: 11px; }
[data-testid="stFileUploader"] { background: #FFF1E8 !important; border: 1.5px dashed #D9A995; border-radius: 16px; padding: 12px; }
[data-testid="stAudioInput"] { background: #FFF1E8 !important; border-radius: 14px; padding: 10px; }

/* Warm four-tab navigation: all tab labels and nested elements stay readable */
.stTabs, div[data-testid="stTabs"] { margin-top: 12px; }
.stTabs [data-baseweb="tab-list"],
div[data-testid="stTabs"] [role="tablist"] {
  display: flex !important;
  flex-wrap: wrap !important;
  align-items: stretch !important;
  gap: 8px !important;
  background: var(--page) !important;
  padding: 8px 0 !important;
  border-bottom: 2px solid var(--line) !important;
}
.stTabs [data-baseweb="tab"],
div[data-testid="stTabs"] button[data-baseweb="tab"],
div[data-testid="stTabs"] button[role="tab"],
div[data-testid="stTabs"] [role="tab"] {
  box-sizing: border-box !important;
  min-height: 48px !important;
  padding: 10px 16px !important;
  margin: 0 !important;
  background: var(--tab-bg) !important;
  color: var(--ink) !important;
  -webkit-text-fill-color: var(--ink) !important;
  border: 1px solid var(--line) !important;
  border-bottom: 4px solid transparent !important;
  border-radius: 10px 10px 0 0 !important;
  font-size: 16px !important;
  font-weight: 700 !important;
  line-height: 1.4 !important;
  opacity: 1 !important;
  box-shadow: none !important;
  text-shadow: none !important;
}
.stTabs [data-baseweb="tab"] *,
div[data-testid="stTabs"] button[data-baseweb="tab"] *,
div[data-testid="stTabs"] button[role="tab"] *,
