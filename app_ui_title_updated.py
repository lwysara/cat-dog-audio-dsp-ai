from pathlib import Path
import hashlib
import io

import joblib
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Warm visual defaults for all Matplotlib DSP / model-insight charts.
plt.rcParams.update({
    "figure.facecolor": "#FFFEFC",
    "axes.facecolor": "#FFFEFC",
    "axes.edgecolor": "#DCC2B4",
    "axes.labelcolor": "#56382F",
    "axes.titlecolor": "#56382F",
    "xtick.color": "#86675B",
    "ytick.color": "#86675B",
    "text.color": "#56382F",
    "grid.color": "#EAD7CB",
    "font.family": "DejaVu Sans",
})
import numpy as np
import pandas as pd
import streamlit as st
from scipy.signal import butter, sosfilt

# ============================================================
# Explainable Audio DSP AI — Phase 1 integration
# Keeps the existing model bundle contract and DSP explanations.
# ============================================================

st.set_page_config(
    page_title="Explainable Audio DSP AI",
    page_icon="🎙️",
    layout="wide",
)

# -------------------- Modern GUI theme --------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Noto+Sans+TC:wght@400;500;600;700;800&display=swap');
:root {
  --page: #FFF8F2;
  --surface: #FFFEFC;
  --surface-soft: #F8EDE5;
  --surface-warm: #FFF0E7;
  --ink: #56382F;
  --muted: #86675B;
  --accent: #D96150;
  --accent-dark: #9C4034;
  --accent-light: #FFE2D7;
  --line: #EAD7CB;
  --shadow: 0 10px 28px rgba(92, 56, 44, .07);
}
html, body, [class*="css"] { font-family: 'DM Sans', 'Noto Sans TC', sans-serif; }
html, body, .stApp, [data-testid="stAppViewContainer"] {
  background: var(--page) !important; color: var(--ink) !important;
}
[data-testid="stHeader"] { background: rgba(255,248,242,.92) !important; }
.block-container { max-width: 1480px; padding: 1.8rem 2rem 3.5rem; }
[data-testid="stMainBlockContainer"] { gap: 1.25rem; }
h1,h2,h3,h4,h5,h6 { color: var(--ink) !important; letter-spacing: -.025em; }
h1 { font-weight: 800 !important; }
h2 { font-weight: 750 !important; }
h3 { font-weight: 700 !important; }
p, label, li, .stMarkdown { color: var(--ink); }
small, [data-testid="stCaptionContainer"] { color: var(--muted) !important; }
hr { border-color: var(--line) !important; margin: 1.25rem 0 !important; }

/* Sidebar: warm espresso navigation */
section[data-testid="stSidebar"] { background: linear-gradient(165deg,#855747 0%,#684438 55%,#52372F 100%) !important; border-right: 1px solid #D7B6A5; }
section[data-testid="stSidebar"] > div { background: transparent !important; }
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"], section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] * {
  color: #FFF8F2 !important;
}
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: #F2DCD0 !important; }
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] code { background:#52372F; color:#FFE6D8 !important; }

/* Hero panel */
.hero-card { position:relative; overflow:hidden; padding:30px 34px; border-radius:24px; background:linear-gradient(115deg,#FFFEFC 0%,#FFF0E7 64%,#F7DFD2 100%); border:1px solid var(--line); box-shadow:var(--shadow); margin:0 0 22px; }
.hero-card:after { content:'♫'; position:absolute; right:30px; top:-28px; font-size:150px; line-height:1; color:rgba(201,101,77,.10); pointer-events:none; }
.hero-kicker { font-size:11px; font-weight:800; letter-spacing:.18em; color:var(--accent-dark) !important; text-transform:uppercase; }
.hero-title { font-size:clamp(28px,3.0vw,44px); font-weight:800; line-height:1.12; color:var(--ink) !important; margin:10px 0 12px; }
.hero-sub { max-width:850px; font-size:16px; line-height:1.7; color:var(--muted) !important; margin:0; }
.section-label { color:var(--accent-dark) !important; font-size:11px; font-weight:800; letter-spacing:.16em; margin:12px 0 4px; }
.dsp-section-title { color:var(--ink) !important; font-size:clamp(32px,3.4vw,46px) !important; font-weight:800 !important; letter-spacing:-.03em !important; line-height:1.2 !important; margin:6px 0 20px !important; }

/* Containers and expanders */
[data-testid="stVerticalBlockBorderWrapper"], [data-testid="stExpander"] { border:1px solid var(--line) !important; border-radius:18px !important; background:rgba(255,254,252,.82) !important; box-shadow:0 5px 18px rgba(92,56,44,.035); }
[data-testid="stExpander"] summary { color:var(--ink) !important; font-weight:700 !important; }
[data-testid="stAlert"] { border-radius:14px !important; border:1px solid var(--line) !important; }
[data-testid="stMetric"] { background:linear-gradient(135deg,#FFFEFC,#F8EDE5) !important; border:1px solid var(--line) !important; padding:18px 20px !important; border-radius:18px !important; box-shadow:var(--shadow); }
[data-testid="stMetricLabel"] { color:var(--muted) !important; font-weight:650 !important; }
