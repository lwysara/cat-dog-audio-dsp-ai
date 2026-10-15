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
:root { --ink:#49352d; --muted:#806d64; --coral:#e77b65; --pink:#f4b7ad; --line:#f0ddd3; }
html, body, [class*="css"] { font-family: Inter, "Noto Sans TC", sans-serif; }
.stApp { background: linear-gradient(135deg,#fffaf4 0%,#fff3ec 52%,#fff7f7 100%); color:var(--ink); }
.block-container { max-width: 1500px; padding-top: 1.5rem; padding-bottom: 3rem; }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#fff0e5 0%,#ffe8e4 100%); border-right:1px solid #f1d5c8; }
[data-testid="stSidebar"] * { color:#594138; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:#74594d; }
h1,h2,h3 { color:var(--ink); letter-spacing:-.025em; }
h1 { font-weight:800 !important; }
[data-testid="stMetric"] { background:linear-gradient(135deg,#ffffff,#fff3ed); border:1px solid var(--line); padding:18px 20px; border-radius:18px; box-shadow:0 8px 24px rgba(133,83,61,.07); }
[data-testid="stMetricLabel"] { color:#806d64 !important; font-weight:600; }
[data-testid="stMetricValue"] { color:#b95e4c !important; font-weight:800; }
div[data-testid="stVerticalBlockBorderWrapper"] { border-radius:18px; }
.stButton > button[kind="primary"] { background:linear-gradient(90deg,#e9856e,#d96b70); color:white; border:0; border-radius:12px; min-height:3rem; font-weight:700; box-shadow:0 7px 18px rgba(217,107,112,.2); }
.stButton > button[kind="primary"]:hover { border:0; filter:brightness(1.04); box-shadow:0 9px 24px rgba(217,107,112,.28); }
.stButton > button, .stDownloadButton > button { border-radius:11px; }
/* Audio source selector buttons: readable text and comfortable touch targets. */
div[data-testid="stButton"] > button {
    min-height: 3.1rem;
    font-weight: 700;
    font-size: 1rem;
    border: 1px solid #e9b5a4;
}
div[data-testid="stButton"] > button[kind="secondary"] {
    background: #fffaf5;
    color: #65483d !important;
}
div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(90deg, #e9856e, #d96b70);
    color: #ffffff !important;
    border: 0;
}
[data-testid="stFileUploader"] { background:#fff7f0; border:1.5px dashed #e9b5a4; border-radius:16px; padding:12px; }
[data-testid="stAudioInput"] { background:#fff0ed; border-radius:14px; padding:10px; }
.stTabs [data-baseweb="tab-list"] { gap:8px; background:#fff; padding:8px; border:1px solid var(--line); border-radius:14px; }
.stTabs [data-baseweb="tab"] { border-radius:10px; font-weight:600; }
.stTabs [aria-selected="true"] { background:#ffe8df; color:#a74f40; }
[data-testid="stDataFrame"], [data-testid="stTable"] { border:1px solid var(--line); border-radius:12px; overflow:hidden; }
hr { border-color:var(--line); }
div[data-testid="stAlert"] { border-radius:12px; }
/* Make the AI workflow expander title warm and lighter, not dark */
[data-testid="stExpander"] details summary,
[data-testid="stExpander"] details summary p,
[data-testid="stExpander"] details summary span {
    color: #c46c56 !important;
    font-weight: 700 !important;
}
[data-testid="stExpander"] details summary svg {
    color: #c46c56 !important;
    fill: #c46c56 !important;
}
.hero-card { padding:25px 28px; border-radius:22px; background:linear-gradient(110deg,#fffefa 0%,#fff0e5 58%,#ffe7e8 100%); border:1px solid #f1d8ca; box-shadow:0 10px 30px rgba(133,83,61,.07); margin-bottom:18px; }
.hero-kicker { font-size:12px; font-weight:800; letter-spacing:.14em; color:#c46c56; text-transform:uppercase; }
.hero-title { font-size:clamp(27px,3vw,42px); font-weight:800; line-height:1.15; color:#49352d; margin:8px 0; }
.hero-sub { font-size:15px; color:#806d64; margin-bottom:0; }
.section-label { color:#bd6551; font-size:12px; font-weight:800; letter-spacing:.12em; }
@media (max-width: 768px) { .block-container { padding-left:1rem; padding-right:1rem; } .hero-card { padding:20px; } }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-card">
  <div class="hero-kicker">AI AUDIO LAB · DSP + MACHINE LEARNING</div>
  <div class="hero-title">Audio DSP AI ＋ Machine Learning</div>
  <p class="hero-sub">Understand every sound. 從聲音訊號、頻譜特徵到 AI 分類，讓每一次分析都有跡可循。</p>
</div>
""", unsafe_allow_html=True)
