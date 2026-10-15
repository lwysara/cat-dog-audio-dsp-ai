from pathlib import Path
import io
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import streamlit as st
from scipy.signal import butter, sosfilt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# ============================================================
# Audio DSP AI + Machine Learning
# Based on the user-provided Cats_Dogs DSP script.
# IMPORTANT: the demo classifier below is trained on synthetic
# feature rows, not on a real labelled cat/dog audio dataset.
# ============================================================

st.set_page_config(
    page_title="Audio DSP AI + Machine Learning",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------- UI theme --------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Noto+Sans+TC:wght@400;500;600;700;800&display=swap');
:root { --ink:#49352d; --muted:#806d64; --coral:#d96f58; --line:#f0ddd3; }
html, body, [class*="css"] { font-family: Inter, "Noto Sans TC", sans-serif; }
.stApp { background: linear-gradient(135deg,#fffaf4 0%,#fff3ec 52%,#fff7f7 100%); color:var(--ink); }
.block-container { max-width: 1500px; padding-top: 1.4rem; padding-bottom: 3rem; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#fff0e5 0%,#ffe8e4 100%); border-right:1px solid #f1d5c8; }
h1,h2,h3 { color:var(--ink); letter-spacing:-.02em; }
[data-testid="stMetric"] { background:linear-gradient(135deg,#fff,#fff3ed); border:1px solid var(--line); padding:14px 18px; border-radius:16px; }
[data-testid="stMetricLabel"] { color:#806d64 !important; }
[data-testid="stMetricValue"] { color:#b95e4c !important; }
.stButton > button[kind="primary"] { background:linear-gradient(90deg,#e9856e,#d96b70); color:white; border:0; border-radius:12px; min-height:2.8rem; font-weight:700; }
.stButton > button { border-radius:11px; }
.stTabs [data-baseweb="tab-list"] { gap:8px; background:#fff; padding:7px; border:1px solid var(--line); border-radius:14px; }
.stTabs [data-baseweb="tab"] { border-radius:10px; font-weight:600; }
.hero-card { padding:26px 30px; border-radius:22px; background:linear-gradient(110deg,#fffefa 0%,#fff0e5 58%,#ffe7e8 100%); border:1px solid #f1d8ca; box-shadow:0 10px 30px rgba(133,83,61,.07); margin-bottom:18px; }
.hero-kicker { font-size:12px; font-weight:800; letter-spacing:.14em; color:#c46c56; text-transform:uppercase; }
.hero-title { font-size:clamp(28px,3.8vw,48px); font-weight:800; line-height:1.15; color:#49352d; margin:8px 0; }
.hero-sub { font-size:16px; color:#806d64; margin-bottom:0; }
.flow-note { color:#bd6551; font-weight:700; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-card">
  <div class="hero-kicker">AI AUDIO LAB · DSP + MACHINE LEARNING</div>
  <div class="hero-title">Audio DSP AI + Machine Learning</div>
  <p class="hero-sub">Understand every sound. 從聲音訊號、頻譜特徵到 AI 分類，讓每一次分析都有跡可循。</p>
</div>
""", unsafe_allow_html=True)

with st.expander("📌 AI 分析流程：22 DSP Features → Random Forest → Explainability", expanded=False):
    st.markdown("""
    **Audio Input → Band-pass Filter → 22 DSP Features → StandardScaler → Random Forest → Prediction + Feature Importance**

    本範例的 Random Forest 使用程式生成的合成特徵資料建立示範模型，並非以真實貓狗音檔訓練。
    若要將結果用於實際辨識，請改用已標註的真實音訊資料重新訓練與驗證模型。
    """)

# -------------------- DSP feature pipeline --------------------
DEFAULT_SR = 22050
FEATURE_KEYS = [
    "RMS", "Peak", "ZCR", "Crest_Factor",
    "Spectral_Centroid", "Spectral_Bandwidth", "Spectral_Rolloff",
    "Spectral_Flatness", "Dominant_Frequency",
] + [f"MFCC_{i}" for i in range(1, 14)]


def bandpass_filter(signal, sr, lowcut=300, highcut=4000, order=5):
    """5th-order Butterworth band-pass filter."""
    if signal is None or len(signal) < 4:
        return signal
    nyquist = sr * 0.5
    low = max(float(lowcut) / nyquist, 1e-5)
    high = min(float(highcut) / nyquist, 0.99)
    if low >= high:
        raise ValueError("高頻切點必須大於低頻切點，且需低於 Nyquist 頻率。")
    return sosfilt(butter(order, [low, high], btype="bandpass", output="sos"), signal)


def extract_dsp_features(signal, sr):
    """Extract 9 time/frequency features plus 13 MFCC means = 22 features."""
    if signal is None or len(signal) == 0:
        raise ValueError("音訊沒有可分析的樣本。")
    rms = float(np.sqrt(np.mean(np.square(signal))))
    peak = float(np.max(np.abs(signal)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=signal)))
    crest_factor = float(peak / (rms + 1e-12))
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=signal, sr=sr)))
    bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=signal, sr=sr)))
    rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=signal, sr=sr)))
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=signal)))
    fft = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), d=1 / sr)
    dominant = float(freqs[np.argmax(fft)]) if len(fft) else 0.0
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
    features = {
        "RMS": rms, "Peak": peak, "ZCR": zcr, "Crest_Factor": crest_factor,
        "Spectral_Centroid": centroid, "Spectral_Bandwidth": bandwidth,
        "Spectral_Rolloff": rolloff, "Spectral_Flatness": flatness,
        "Dominant_Frequency": dominant,
    }
    features.update({f"MFCC_{i+1}": float(np.mean(mfccs[i])) for i in range(13)})
    return features


@st.cache_resource
def build_demo_model():
    """Create a reproducible demo classifier from synthetic feature rows."""
    rng = np.random.default_rng(42)
    rows, labels = [], []
    for label in ("cat", "dog"):
        for _ in range(250):
            is_cat = label == "cat"
            centroid = rng.normal(2200, 400) if is_cat else rng.normal(1200, 300)
            zcr = rng.normal(0.08, 0.02) if is_cat else rng.normal(0.14, 0.03)
            flatness = rng.normal(0.015, 0.005) if is_cat else rng.normal(0.035, 0.01)
            dominant = rng.normal(1100, 250) if is_cat else rng.normal(450, 150)
            row = {
                "RMS": rng.uniform(0.02, 0.12) if is_cat else rng.uniform(0.05, 0.25),
                "Peak": rng.uniform(0.15, 0.6) if is_cat else rng.uniform(0.4, 0.95),
                "ZCR": max(0.01, zcr),
                "Crest_Factor": rng.uniform(3, 6) if is_cat else rng.uniform(4, 9),
                "Spectral_Centroid": max(300, centroid),
                "Spectral_Bandwidth": rng.normal(1800, 300) if is_cat else rng.normal(1300, 250),
                "Spectral_Rolloff": centroid * (1.8 if is_cat else 1.5),
                "Spectral_Flatness": max(0.001, flatness),
                "Dominant_Frequency": max(100, dominant),
            }
            for m in range(1, 14):
                row[f"MFCC_{m}"] = rng.normal(-15 + m * 2, 8) if is_cat else rng.normal(-5 - m * 1.5, 10)
            rows.append([row[k] for k in FEATURE_KEYS])
            labels.append(label)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(rows)
    clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    clf.fit(X_scaled, labels)
    return clf, scaler


model, scaler = build_demo_model()

# -------------------- Sidebar controls --------------------
st.sidebar.header("⚙️ 音訊輸入與 DSP 設定")
source_mode = st.sidebar.radio("音訊來源", ["File Upload", "Microphone"], horizontal=True)
audio_bytes = None
source_name = None

if source_mode == "File Upload":
    uploaded = st.sidebar.file_uploader(
        "上傳音訊檔案", type=["wav", "mp3", "flac", "m4a", "ogg", "aiff"]
    )
    if uploaded is not None:
        audio_bytes = uploaded.getvalue()
        source_name = uploaded.name
else:
    if hasattr(st, "audio_input"):
        recording = st.sidebar.audio_input("錄製音訊")
        if recording is not None:
            audio_bytes = recording.getvalue()
            source_name = "microphone_recording.wav"
    else:
        st.sidebar.warning("目前 Streamlit 版本不支援麥克風元件，請升級 Streamlit。")

enable_filter = st.sidebar.checkbox("啟用巴特沃斯帶通濾波器", value=True)
col1, col2 = st.sidebar.columns(2)
lowcut = col1.number_input("低頻切點 (Hz)", min_value=1, max_value=10000, value=300, step=50)
highcut = col2.number_input("高頻切點 (Hz)", min_value=2, max_value=20000, value=4000, step=100)
st.sidebar.caption("建議切點：300–4000 Hz。高頻切點必須低於取樣率的一半。")
demo_clicked = st.sidebar.button("🎲 生成模擬貓聲／狗聲樣本", use_container_width=True)

if demo_clicked:
    demo_kind = np.random.choice(["cat", "dog"])
    sr_demo = DEFAULT_SR
    t = np.arange(int(sr_demo * 1.5)) / sr_demo
    if demo_kind == "cat":
        instantaneous_freq = np.linspace(900, 1400, len(t))
        signal_demo = 0.3 * np.sin(2 * np.pi * instantaneous_freq * t) * np.sin(np.pi * t / 1.5) ** 2
    else:
        signal_demo = np.zeros_like(t)
        for start in (0.2, 0.7):
            i0 = int(start * sr_demo)
            n = int(0.25 * sr_demo)
            tb = np.arange(n) / sr_demo
            signal_demo[i0:i0+n] += 0.7 * np.sin(2 * np.pi * 350 * tb) * np.exp(-15 * tb)
    st.session_state["demo_audio"] = signal_demo.astype(np.float32).tobytes()
    st.session_state["demo_sr"] = sr_demo
    st.session_state["demo_kind"] = demo_kind
    st.session_state["demo_source_name"] = f"synthetic_{demo_kind}.wav"

if audio_bytes is None and "demo_audio" in st.session_state:
    audio_bytes = st.session_state["demo_audio"]
    source_name = st.session_state.get("demo_source_name", "synthetic_demo.wav")
    audio_is_demo = True
else:
    audio_is_demo = False

# -------------------- Analysis --------------------
st.markdown("### 🎙️ Audio Input")
if audio_bytes is None:
    st.info("請在左側選擇 File Upload 或 Microphone，或按「生成模擬貓聲／狗聲樣本」體驗分析。")
else:
    st.caption(f"音訊來源：{source_name}")
    st.audio(audio_bytes)
    if st.button("▶️ 開始分析", type="primary", use_container_width=True):
        try:
            with st.spinner("正在執行 DSP 與 AI 分析…"):
                y_raw, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
                if len(y_raw) < int(0.25 * sr):
                    raise ValueError("音訊太短，請使用至少約 0.25 秒的音訊。")
                # Keep analysis sample rate consistent for stable filtering/feature extraction.
                if sr != DEFAULT_SR:
                    y_raw = librosa.resample(y_raw, orig_sr=sr, target_sr=DEFAULT_SR)
                    sr = DEFAULT_SR
                if enable_filter:
                    y = bandpass_filter(y_raw, sr, lowcut, highcut)
                else:
                    y = y_raw
                features = extract_dsp_features(y, sr)
                X = pd.DataFrame([[features[k] for k in FEATURE_KEYS]], columns=FEATURE_KEYS)
                X_scaled = scaler.transform(X)
                pred = model.predict(X_scaled)[0]
                proba = model.predict_proba(X_scaled)[0]
                prob_map = {str(c): float(p) for c, p in zip(model.classes_, proba)}
                st.session_state["analysis_result"] = {
                    "source_name": source_name,
                    "y_raw": y_raw,
                    "y": y,
                    "sr": sr,
                    "features": features,
                    "pred": pred,
                    "prob_map": prob_map,
                    "enable_filter": enable_filter,
                    "demo": audio_is_demo,
                }
        except Exception as exc:
            st.error(f"音訊分析失敗：{exc}")

result = st.session_state.get("analysis_result")
if result and result.get("source_name") == source_name:
    y_raw = result["y_raw"]
    y = result["y"]
    sr = result["sr"]
    features = result["features"]
    pred = result["pred"]
    prob_map = result["prob_map"]
    st.success("DSP 與 AI 分析完成")
    if result.get("demo"):
        st.warning("目前使用的是模擬音訊與合成資料訓練的示範模型；預測僅供程式流程展示，不代表真實貓狗辨識準確度。")
    elif source_name:
        st.info("此版本使用合成特徵訓練的示範模型。若要取得可靠的真實分類結果，需使用真實標註資料訓練模型。")
    a, b, c = st.columns(3)
    a.metric("AI 預測結果", "🐱 Cat／貓叫聲" if pred == "cat" else "🐶 Dog／狗叫聲")
    b.metric("Cat 機率（示範）", f"{prob_map.get('cat', 0)*100:.1f}%")
    c.metric("Dog 機率（示範）", f"{prob_map.get('dog', 0)*100:.1f}%")
    st.progress(float(np.clip(prob_map.get(pred, 0), 0, 1)), text="示範模型對預測類別的估計機率")

    tab1, tab2, tab3 = st.tabs(["🎵 訊號與時頻視覺化", "📊 22 項 DSP 特徵", "🧠 AI 模型解釋"])
    with tab1:
        st.subheader("音訊播放與時域波形")
        time_axis = np.arange(len(y_raw)) / sr
        fig, axes = plt.subplots(3, 1, figsize=(11, 8))
        axes[0].plot(time_axis, y_raw, alpha=0.7, label="Raw waveform")
        if result["enable_filter"]:
            axes[0].plot(time_axis, y, alpha=0.8, label="Filtered waveform")
        axes[0].set_title("Time-Domain Waveform")
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Amplitude")
        axes[0].legend()
        axes[0].grid(True, alpha=0.25)

        fft = np.abs(np.fft.rfft(y))
        freqs = np.fft.rfftfreq(len(y), d=1 / sr)
        mag_db = 20 * np.log10(fft + 1e-12)
        axes[1].plot(freqs, mag_db)
        axes[1].axvline(features["Dominant_Frequency"], color="#d96f58", linestyle="--",
                        label=f"Dominant: {features['Dominant_Frequency']:.1f} Hz")
        axes[1].set_title("FFT Frequency Spectrum")
        axes[1].set_xlabel("Frequency (Hz)")
        axes[1].set_ylabel("Magnitude (dB)")
        axes[1].set_xlim(0, sr / 2)
        axes[1].legend()
        axes[1].grid(True, alpha=0.25)

        D = librosa.stft(y, n_fft=1024, hop_length=256)
        D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        img = librosa.display.specshow(D_db, sr=sr, hop_length=256, x_axis="time", y_axis="hz", ax=axes[2], cmap="magma")
        axes[2].set_title("STFT Spectrogram")
        fig.colorbar(img, ax=axes[2], format="%+2.0f dB")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with tab2:
        st.subheader("DSP Feature Values")
        feature_df = pd.DataFrame({"Feature": list(features.keys()), "Value": list(features.values())})
        st.dataframe(feature_df, use_container_width=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("RMS", f"{features['RMS']:.5f}")
        m2.metric("Peak", f"{features['Peak']:.5f}")
        m3.metric("ZCR", f"{features['ZCR']:.5f}")
        m4.metric("Dominant Frequency", f"{features['Dominant_Frequency']:.1f} Hz")
        mfcc_values = [features[f"MFCC_{i}"] for i in range(1, 14)]
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.bar([f"MFCC_{i}" for i in range(1, 14)], mfcc_values)
        ax.set_title("13 MFCC Mean Coefficients")
        ax.set_ylabel("Mean coefficient")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with tab3:
        st.subheader("Random Forest Feature Importance")
        imp_df = pd.DataFrame({
            "Feature": FEATURE_KEYS,
            "Importance": model.feature_importances_,
        }).sort_values("Importance", ascending=True)
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.barh(imp_df["Feature"], imp_df["Importance"])
        ax.set_xlabel("Relative Importance")
        ax.set_title("Random Forest DSP Feature Importance")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Feature importance 顯示示範模型使用特徵的相對程度，不代表物理因果關係或真實資料上的辨識效能。")
