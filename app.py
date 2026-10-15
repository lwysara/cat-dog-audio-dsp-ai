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
    --audio-tab-bg: #F8EDE5;
    --audio-active-bg: #FFE5D9;
    --audio-accent: #D95F50;
    --audio-text: #633D32;
    --audio-muted: #76584D;
    --audio-border: #E8CFC1;
}
.stApp { background-color: var(--audio-bg); }

/* Main tab bar */
.stTabs { margin-top: 12px; }
.stTabs [data-baseweb="tab-list"] {
    display: flex; flex-wrap: wrap; align-items: stretch; gap: 8px;
    background-color: var(--audio-bg); padding: 8px 0 0 0;
    border-bottom: 2px solid var(--audio-border);
}
.stTabs button[data-baseweb="tab"] {
    min-height: 54px; padding: 12px 18px;
    background-color: var(--audio-tab-bg) !important;
    color: var(--audio-text) !important;
    border: 1px solid var(--audio-border) !important;
    border-bottom: 4px solid transparent !important;
    border-radius: 10px 10px 0 0;
    font-size: 17px !important; font-weight: 700 !important;
    line-height: 1.5; opacity: 1 !important;
    transition: background-color .2s ease, border-color .2s ease;
}
.stTabs button[data-baseweb="tab"] p,
.stTabs button[data-baseweb="tab"] span {
    color: var(--audio-text) !important; font-size: 17px !important;
    font-weight: 700 !important; opacity: 1 !important;
}
.stTabs button[data-baseweb="tab"]:hover {
    background-color: #F4DCCF !important; border-color: #D9AFA0 !important;
}
.stTabs button[data-baseweb="tab"][aria-selected="true"] {
    background-color: var(--audio-active-bg) !important;
    color: #8E382D !important; border: 1px solid #E8B5A5 !important;
    border-bottom: 4px solid var(--audio-accent) !important;
}
.stTabs button[data-baseweb="tab"][aria-selected="true"] p,
.stTabs button[data-baseweb="tab"][aria-selected="true"] span {
    color: #8E382D !important; font-weight: 800 !important; opacity: 1 !important;
}
.stTabs button[data-baseweb="tab"]:focus-visible {
    outline: 3px solid #B85D4E !important; outline-offset: 3px;
}
.stTabs [data-baseweb="tab-panel"] {
    padding: 22px 10px 12px 10px; color: var(--audio-text);
}
.stTabs [data-baseweb="tab-panel"] h1,
.stTabs [data-baseweb="tab-panel"] h2,
.stTabs [data-baseweb="tab-panel"] h3 {
    color: var(--audio-text) !important; font-weight: 750 !important;
}
.stTabs [data-baseweb="tab-panel"] p,
.stTabs [data-baseweb="tab-panel"] label { color: var(--audio-text); }

/* General readable text and cards */
[data-testid="stMetricLabel"], [data-testid="stMetricValue"],
[data-testid="stMarkdownContainer"] p { color: var(--audio-text); }
div[data-testid="stMetric"] {
    background: #FFFDFB; border: 1px solid var(--audio-border);
    padding: 14px; border-radius: 12px;
}
@media (max-width: 640px) {
    .stTabs [data-baseweb="tab-list"] { gap: 5px; }
    .stTabs button[data-baseweb="tab"] { min-height: 48px; padding: 9px 10px; }
    .stTabs button[data-baseweb="tab"],
    .stTabs button[data-baseweb="tab"] p,
    .stTabs button[data-baseweb="tab"] span { font-size: 15px !important; }
}
</style>
""", unsafe_allow_html=True)

st.title("🐱🐶 Cat & Dog Audio DSP + AI")
