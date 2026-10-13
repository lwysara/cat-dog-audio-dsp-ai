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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Noto+Sans+TC:wght@400;500;600;700;800&display=swap');
:root { --ink:#14213d; --muted:#65738b; --blue:#3978f6; --violet:#7c5cff; --line:#e5ebf5; }
html, body, [class*="css"] { font-family: Inter, "Noto Sans TC", sans-serif; }
.stApp { background: linear-gradient(135deg,#f7f9ff 0%,#f5f7fc 52%,#f9f7ff 100%); color:var(--ink); }
.block-container { max-width: 1500px; padding-top: 1.5rem; padding-bottom: 3rem; }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#172642 0%,#111c32 100%); border-right:1px solid #263957; }
[data-testid="stSidebar"] * { color:#eef4ff; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:#d0dcf1; }
h1,h2,h3 { color:var(--ink); letter-spacing:-.035em; }
h1 { font-weight:800 !important; }
[data-testid="stMetric"] { background:linear-gradient(135deg,#ffffff,#f4f7ff); border:1px solid var(--line); padding:18px 20px; border-radius:18px; box-shadow:0 8px 24px rgba(36,58,100,.055); }
[data-testid="stMetricLabel"] { color:#65738b !important; font-weight:600; }
[data-testid="stMetricValue"] { color:#1d3e88 !important; font-weight:800; }
div[data-testid="stVerticalBlockBorderWrapper"] { border-radius:18px; }
.stButton > button[kind="primary"] { background:linear-gradient(90deg,#7165f5,#3978f6); border:0; border-radius:12px; min-height:3rem; font-weight:700; box-shadow:0 7px 18px rgba(67,105,245,.2); }
.stButton > button[kind="primary"]:hover { border:0; filter:brightness(1.04); box-shadow:0 9px 24px rgba(67,105,245,.28); }
.stButton > button, .stDownloadButton > button { border-radius:11px; }
[data-testid="stFileUploader"] { background:#f1f5ff; border:1.5px dashed #9bb6ff; border-radius:16px; padding:12px; }
[data-testid="stAudioInput"] { background:#fff5f6; border-radius:14px; padding:10px; }
.stTabs [data-baseweb="tab-list"] { gap:8px; background:#fff; padding:8px; border:1px solid var(--line); border-radius:14px; }
.stTabs [data-baseweb="tab"] { border-radius:10px; font-weight:600; }
.stTabs [aria-selected="true"] { background:#edf3ff; color:#285ee8; }
[data-testid="stDataFrame"], [data-testid="stTable"] { border:1px solid var(--line); border-radius:12px; overflow:hidden; }
hr { border-color:var(--line); }
div[data-testid="stAlert"] { border-radius:12px; }
.hero-card { padding:25px 28px; border-radius:22px; background:linear-gradient(110deg,#ffffff 0%,#eef4ff 65%,#f5efff 100%); border:1px solid #e2e9fa; box-shadow:0 10px 30px rgba(42,63,110,.06); margin-bottom:18px; }
.hero-kicker { font-size:12px; font-weight:800; letter-spacing:.14em; color:#5578db; text-transform:uppercase; }
.hero-title { font-size:clamp(27px,3vw,42px); font-weight:800; line-height:1.15; color:#14213d; margin:8px 0; }
.hero-sub { font-size:15px; color:#65738b; margin-bottom:0; }
.section-label { color:#315fce; font-size:12px; font-weight:800; letter-spacing:.12em; }
@media (max-width: 768px) {
  .block-container { padding-left:1rem; padding-right:1rem; }
  .hero-card { padding:20px; }
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-card">
  <div class="hero-kicker">AI AUDIO LAB · DSP + MACHINE LEARNING</div>
  <div class="hero-title">Explainable Audio DSP AI</div>
  <p class="hero-sub">Understand every sound. 從聲音訊號、頻譜特徵到 AI 分類，讓每一次分析都有跡可循。</p>
</div>
""", unsafe_allow_html=True)

with st.expander("📌 AI 分析流程：22 DSP Features → Random Forest → Explainability", expanded=True):
    st.markdown(
        """
**Audio Input → DSP Processing → 22 DSP Features**  
↓  
**Cat / Dog Random Forest**（僅在 `Cat / Dog Classification` 模式執行）  
↓  
**Prediction + DSP Plots + Explainability**

分類模式會使用已載入模型與對應的 StandardScaler 進行預測；
`Audio DSP Analysis Only` 模式只執行 DSP 分析，不會執行 Cat/Dog 分類。
"""
    )

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "cat_dog_model.joblib"
FEATURE_DATA_PATH = ROOT / "data" / "feature_dataset.csv"

DEFAULT_SR = 22050
DEFAULT_LOWCUT, DEFAULT_HIGHCUT = 300, 4000
AUDIO_EXTENSIONS = ["wav", "mp3", "flac", "m4a", "ogg", "aiff", "aif"]

FEATURES = [
    "RMS",
    "Peak",
    "ZCR",
    "Crest_Factor",
    "Spectral_Centroid",
    "Spectral_Bandwidth",
    "Spectral_Rolloff",
    "Spectral_Flatness",
    "Dominant_Frequency",
] + [f"MFCC_{i}" for i in range(1, 14)]


def bandpass(y, sr, low=DEFAULT_LOWCUT, high=DEFAULT_HIGHCUT):
    """Apply the same band-pass filtering approach as the existing app."""
    if len(y) == 0:
        return y

    nyquist = sr / 2.0
    low_norm = max(float(low) / nyquist, 1e-5)
    high_norm = min(float(high) / nyquist, 0.99)

    # Avoid invalid filter cutoffs for unusual sample rates/settings.
    if low_norm >= high_norm:
        return y

    sos = butter(
        5,
        [low_norm, high_norm],
        btype="bandpass",
        output="sos",
    )
    return sosfilt(sos, y)


def extract_features(y, sr):
    """Extract the same 22 DSP features used by the saved model."""
    if len(y) == 0:
        raise ValueError("音訊沒有可分析的樣本。")

    rms = float(np.sqrt(np.mean(y ** 2)))
    peak = float(np.max(np.abs(y)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
    crest = float(peak / (rms + 1e-12))

    magnitude = np.abs(np.fft.rfft(y))
    frequencies = np.fft.rfftfreq(len(y), d=1 / sr)
    dominant = (
        float(frequencies[np.argmax(magnitude)])
        if len(magnitude)
        else 0.0
    )

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    features = {
        "RMS": rms,
        "Peak": peak,
        "ZCR": zcr,
        "Crest_Factor": crest,
        "Spectral_Centroid": centroid,
        "Spectral_Bandwidth": bandwidth,
        "Spectral_Rolloff": rolloff,
        "Spectral_Flatness": flatness,
        "Dominant_Frequency": dominant,
    }
    features.update({
        f"MFCC_{i + 1}": float(np.mean(mfcc[i]))
        for i in range(13)
    })
    return features


@st.cache_resource(show_spinner=False)
def load_model_bundle(model_path_string):
    """Load the existing joblib bundle without changing its schema."""
    model_path = Path(model_path_string)
    if not model_path.exists():
        return None
    return joblib.load(model_path)


@st.cache_data(show_spinner=False)
def load_feature_dataset(csv_path_string):
    csv_path = Path(csv_path_string)
    if not csv_path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(csv_path)
        if "Label" in df.columns and all(c in df.columns for c in FEATURES):
            return df
    except Exception:
        pass

    return pd.DataFrame()


def make_barplot(data, value_col, title, n=15):
    plot_data = data.head(n).sort_values(value_col)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(plot_data["Feature"], plot_data[value_col])
    ax.set_title(title)
    ax.set_xlabel(value_col)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    return fig


# -------------------- Model loading --------------------
bundle = None
model = None
scaler = None
model_load_error = None

try:
    bundle = load_model_bundle(str(MODEL_PATH))
    if bundle is not None:
        # These keys match the existing app.py / train_model.py bundle.
        model = bundle.get("model")
        scaler = bundle.get("scaler")
        if model is None or scaler is None:
            model_load_error = (
                "模型檔存在，但找不到預期的 'model' 或 'scaler' 欄位。"
            )
            bundle = None
            model = None
            scaler = None
except Exception as exc:
    model_load_error = f"模型載入失敗：{exc}"

feature_names = (
    list(bundle.get("feature_keys", FEATURES))
    if bundle is not None
    else FEATURES.copy()
)
sr_model = int(bundle.get("sample_rate", DEFAULT_SR)) if bundle else DEFAULT_SR
lowcut = float(bundle.get("lowcut", DEFAULT_LOWCUT)) if bundle else DEFAULT_LOWCUT
highcut = float(bundle.get("highcut", DEFAULT_HIGHCUT)) if bundle else DEFAULT_HIGHCUT

train_df = load_feature_dataset(str(FEATURE_DATA_PATH))
if train_df.empty and bundle and bundle.get("training_rows"):
    try:
        train_df = pd.DataFrame(bundle["training_rows"])
    except Exception:
        train_df = pd.DataFrame()

# Sidebar settings and model metadata
st.sidebar.markdown("""
<div style="padding:8px 0 20px 0;">
  <div style="font-size:12px;letter-spacing:.16em;font-weight:800;color:#7dd3fc;">● AUDIO INTELLIGENCE</div>
  <div style="font-size:23px;line-height:1.2;font-weight:800;color:#ffffff;margin-top:8px;">Cat & Dog<br>Audio DSP AI</div>
  <div style="font-size:12px;color:#b9c8e2;margin-top:8px;">Signal · Features · Prediction</div>
</div>
""", unsafe_allow_html=True)
st.sidebar.header("⚙️ 模型設定")
st.sidebar.write(f"模型檔案：`{MODEL_PATH.relative_to(ROOT)}`")
st.sidebar.write(f"模型取樣率：{sr_model} Hz")
st.sidebar.write(f"特徵濾波範圍：{lowcut:g}–{highcut:g} Hz")
st.sidebar.caption(
    "預測機率是模型對單筆音訊的估計，不等於整體模型準確率。"
)
if bundle is None:
    st.sidebar.warning("尚未成功載入模型；DSP Analysis Only 模式仍可使用。")
    if model_load_error:
        st.sidebar.caption(model_load_error)

# -------------------- Phase 1 input controls --------------------
st.markdown('<div class="section-label">01 / AUDIO SOURCE</div>', unsafe_allow_html=True)
st.subheader("🎙️ Audio Input")

input_source = st.radio(
    "選擇音訊來源",
    options=["File Upload", "Microphone"],
    index=0,
    horizontal=True,
    help="預設為檔案上傳；也可以切換成麥克風錄音。",
)

uploaded_file = None
recorded_audio = None

if input_source == "File Upload":
    uploaded_file = st.file_uploader(
        "📁 上傳音訊檔案",
        type=AUDIO_EXTENSIONS,
        help="支援 WAV、MP3、FLAC、M4A、OGG、AIFF 等格式；實際解碼能力取決於部署環境。",
        key="phase1_audio_file_uploader",
    )
    selected_source = uploaded_file
else:
    if hasattr(st, "audio_input"):
        recorded_audio = st.audio_input(
            "🎙️ 錄製音訊",
            help="請允許瀏覽器使用麥克風；錄音完成後再按下開始分析。",
            key="phase1_audio_recorder",
        )
        selected_source = recorded_audio
    else:
        selected_source = None
        st.error(
            "目前的 Streamlit 版本不支援 st.audio_input。"
            "請在 requirements.txt 將 Streamlit 更新至支援此元件的版本。"
        )

analysis_mode = st.selectbox(
    "選擇分析模式",
    options=[
        "Cat / Dog Classification",
        "Audio DSP Analysis Only",
    ],
    index=0,
    help=(
        "Cat / Dog Classification 會執行模型預測與既有解釋功能；"
        "Audio DSP Analysis Only 只顯示音訊 DSP 分析。"
    ),
)

raw_bytes = selected_source.getvalue() if selected_source is not None else None
source_name = (
    getattr(selected_source, "name", "microphone_recording.wav")
    if selected_source is not None
    else None
)

if raw_bytes:
    st.audio(raw_bytes)
    st.caption(f"目前音訊來源：{input_source} · 檔案名稱：{source_name}")

# Only analyse after an explicit button click. Keep results on later reruns
# as long as the audio, analysis mode, and filter settings have not changed.
current_analysis_key = None
if raw_bytes:
    fingerprint = hashlib.sha256(raw_bytes).hexdigest()
    current_analysis_key = (
        f"{fingerprint}|{analysis_mode}|{sr_model}|{lowcut}|{highcut}"
    )

start_clicked = st.button(
    "▶️ 開始分析",
    type="primary",
    use_container_width=True,
)

if start_clicked:
    if not raw_bytes:
        st.warning("請先上傳音訊或完成錄音，再按「開始分析」。")
    elif analysis_mode == "Cat / Dog Classification" and bundle is None:
        st.error(
            "目前無法進行 Cat / Dog Classification，因為模型沒有成功載入。"
            "請確認 models/cat_dog_model.joblib 存在且格式正確。"
        )
    else:
        st.session_state["phase1_last_analysis_key"] = current_analysis_key

last_analysis_key = st.session_state.get("phase1_last_analysis_key")
should_show_results = (
    bool(raw_bytes)
    and current_analysis_key is not None
    and last_analysis_key == current_analysis_key
)

if not raw_bytes:
    st.info("請先選擇音訊來源並提供音訊，再按「開始分析」。")
elif not should_show_results:
    st.info("音訊已就緒。確認分析模式後，請按「▶️ 開始分析」執行分析。")

if should_show_results:
    try:
        with st.spinner("正在分析音訊，請稍候..."):
            y_raw, sr = librosa.load(
                io.BytesIO(raw_bytes),
                sr=sr_model,
                mono=True,
            )

            if len(y_raw) < int(0.25 * sr):
                raise ValueError(
                    "音訊太短，請提供至少約 0.25 秒的音訊。"
                )

            y = bandpass(y_raw, sr, lowcut, highcut)
            feats = extract_features(y, sr)

            # -------------------- Cat/Dog model prediction --------------------
            pred = None
            probs = None
            classes = []
            prob_map = {}
            confidence = None
            x = None

            if analysis_mode == "Cat / Dog Classification":
                if model is None or scaler is None or bundle is None:
                    raise RuntimeError("模型或 StandardScaler 尚未成功載入。")

                missing_features = [name for name in feature_names if name not in feats]
                if missing_features:
                    raise ValueError(
                        "目前音訊特徵無法符合模型所需欄位："
                        + ", ".join(missing_features)
                    )

                # Preserve the exact feature ordering stored with the model.
                x = pd.DataFrame(
                    [[feats[name] for name in feature_names]],
                    columns=feature_names,
                )
                x_scaled = scaler.transform(x)
                pred = str(model.predict(x_scaled)[0]).lower()

                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(x_scaled)[0]
                    classes = [str(c).lower() for c in model.classes_]
                    prob_map = {
                        classes[i]: float(probs[i])
                        for i in range(len(classes))
                    }
                    confidence = prob_map.get(
                        pred,
                        float(np.max(probs)),
                    )

        # -------------------- Results --------------------
        st.success("分析完成。")
        st.markdown('<div class="section-label">02 / SIGNAL EXPLORER</div>', unsafe_allow_html=True)
        st.subheader("Audio DSP 分析")

        if analysis_mode == "Cat / Dog Classification":
            st.subheader("AI 預測結果")
            c1, c2, c3 = st.columns(3)
            label_text = (
                "🐱 Cat／貓叫聲" if pred == "cat"
                else "🐶 Dog／狗叫聲" if pred == "dog"
                else str(pred).upper()
            )
            c1.metric("辨識結果", label_text)
            c2.metric("Cat 機率", f"{prob_map.get('cat', 0) * 100:.1f}%")
            c3.metric("Dog 機率", f"{prob_map.get('dog', 0) * 100:.1f}%")

            if confidence is not None:
                st.progress(
                    float(np.clip(confidence, 0, 1)),
                    text=f"此音訊的預測類別機率：{confidence * 100:.1f}%",
                )
            else:
                st.warning("此模型沒有 predict_proba()，因此無法顯示分類機率。")

        t1, t2, t3, t4 = st.tabs(
            ["🌊 Waveform", "📈 FFT", "🔥 Spectrogram", "🎼 MFCC"]
        )

        with t1:
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(
                np.arange(len(y_raw)) / sr,
                y_raw,
                linewidth=0.65,
                label="Raw",
            )
            ax.plot(
                np.arange(len(y)) / sr,
                y,
                linewidth=0.65,
                label="Bandpass filtered",
            )
            ax.set(
                xlabel="Time (s)",
                ylabel="Amplitude",
                title="Time-Domain Waveform",
            )
            ax.legend()
            ax.grid(alpha=0.2)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with t2:
            spectrum = np.abs(np.fft.rfft(y_raw))
            freqs = np.fft.rfftfreq(len(y_raw), d=1 / sr)
            db = 20 * np.log10(spectrum / (spectrum.max() + 1e-12) + 1e-12)
            dominant = float(freqs[np.argmax(spectrum)])

            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(freqs, db, linewidth=0.65)
            ax.axvline(
                dominant,
                linestyle="--",
                label=f"Dominant = {dominant:.1f} Hz",
            )
            ax.set_xlim(0, min(10000, sr / 2))
            ax.set(
                xlabel="Frequency (Hz)",
                ylabel="Magnitude (dB)",
                title="FFT Spectrum",
            )
            ax.legend()
            ax.grid(alpha=0.2)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with t3:
            D = librosa.amplitude_to_db(
                np.abs(librosa.stft(y)),
                ref=np.max,
            )
            fig, ax = plt.subplots(figsize=(10, 4))
            img = librosa.display.specshow(
                D,
                sr=sr,
                x_axis="time",
                y_axis="hz",
                ax=ax,
                cmap="magma",
            )
            ax.set_ylim(0, min(10000, sr / 2))
            ax.set_title("STFT Spectrogram")
            fig.colorbar(img, ax=ax, format="%+2.0f dB")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with t4:
            mfcc_matrix = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            fig, ax = plt.subplots(figsize=(10, 4))
            img = librosa.display.specshow(
                mfcc_matrix,
                x_axis="time",
                ax=ax,
                cmap="coolwarm",
            )
            ax.set_title("MFCC")
            ax.set_ylabel("MFCC coefficient")
            fig.colorbar(img, ax=ax)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.markdown('<div class="section-label">03 / FEATURE ENGINEERING</div>', unsafe_allow_html=True)
        st.subheader("DSP Feature Values")
        feature_table = pd.DataFrame({
            "Feature": list(feats.keys()),
            "Value": list(feats.values()),
        })
        st.dataframe(feature_table, use_container_width=True)

        # -------------------- Existing explainability features --------------------
        if analysis_mode == "Cat / Dog Classification" and bundle is not None:
            if hasattr(model, "feature_importances_"):
                st.markdown('<div class="section-label">04 / MODEL EXPLAINABILITY</div>', unsafe_allow_html=True)
                st.subheader("Global Feature Importance")
                importances = np.asarray(model.feature_importances_)
                if len(importances) == len(feature_names):
                    importance = pd.DataFrame({
                        "Feature": feature_names,
                        "Importance": importances,
                    }).sort_values("Importance", ascending=False)

                    fig = make_barplot(
                        importance,
                        "Importance",
                        "Random Forest Global Feature Importance",
                        n=15,
                    )
                    st.pyplot(fig)
                    plt.close(fig)

                    with st.expander("查看完整 Global Feature Importance 表"):
                        st.dataframe(importance, use_container_width=True)
                    st.caption(
                        "Global Feature Importance 描述模型整體使用特徵的程度，"
                        "不代表物理因果關係。"
                    )
                else:
                    st.warning(
                        "模型的 feature_importances_ 長度與 feature_keys 不一致；"
                        "為避免錯誤對應，略過此圖。"
                    )

            perm_rows = bundle.get("permutation_importance")
            if perm_rows:
                try:
                    perm = pd.DataFrame(perm_rows)
                    if {"Feature", "Importance"}.issubset(perm.columns):
                        perm = perm.sort_values("Importance", ascending=False)
                        st.subheader("⑤ Permutation Importance")
                        fig = make_barplot(
                            perm,
                            "Importance",
                            "Permutation Importance",
                            n=15,
                        )
                        st.pyplot(fig)
                        plt.close(fig)

                        with st.expander("查看完整 Permutation Importance 表"):
                            st.dataframe(perm, use_container_width=True)
                except Exception as exc:
                    st.warning(f"無法顯示 Permutation Importance：{exc}")

            st.subheader("⑥ Cat vs Dog Feature Distribution")
            required_distribution_cols = (
                {"Label"} | set(feature_names)
            )
            if (
                not train_df.empty
                and required_distribution_cols.issubset(train_df.columns)
            ):
                default_feature = (
                    "Spectral_Centroid"
                    if "Spectral_Centroid" in feature_names
                    else feature_names[0]
                )
                selected_feature = st.selectbox(
                    "選擇 DSP Feature 比較 Cat 與 Dog",
                    feature_names,
                    index=feature_names.index(default_feature),
                )

                labels = train_df["Label"].astype(str).str.lower()
                catv = train_df.loc[
                    labels == "cat", selected_feature
                ].dropna()
                dogv = train_df.loc[
                    labels == "dog", selected_feature
                ].dropna()

                fig, ax = plt.subplots(figsize=(9, 5))
                if len(catv):
                    ax.hist(catv, bins=30, alpha=0.55, label="Cat")
                    ax.axvline(
                        catv.mean(),
                        linestyle="--",
                        label="Cat Mean",
                    )
                if len(dogv):
                    ax.hist(dogv, bins=30, alpha=0.55, label="Dog")
                    ax.axvline(
                        dogv.mean(),
                        linestyle="--",
                        label="Dog Mean",
                    )
                ax.set_xlabel(selected_feature)
                ax.set_ylabel("Samples")
                ax.set_title(f"Cat vs Dog Distribution: {selected_feature}")
                ax.legend()
                ax.grid(alpha=0.25)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                a, b, c = st.columns(3)
                a.metric("Cat samples", len(catv))
                b.metric("Dog samples", len(dogv))
                gap = (
                    abs(float(catv.mean()) - float(dogv.mean()))
                    if len(catv) and len(dogv)
                    else float("nan")
                )
                c.metric(
                    "Absolute Mean Gap",
                    f"{gap:.4g}" if np.isfinite(gap) else "N/A",
                )
            else:
                st.info(
                    "尚未找到可用的 data/feature_dataset.csv 或訓練資料列。"
                    "預測與 DSP 分析仍可使用；若要顯示此分布圖，"
                    "請確認 CSV 含 Label 與全部模型特徵欄位。"
                )

            st.subheader("⑦ Local Feature Impact")
            st.write(
                "逐一將單一特徵替換為訓練資料中位數，觀察目前預測類別機率的變化。"
                "這是近似模型解釋，不是因果證明。"
            )
            medians = bundle.get("training_medians", {})
            if (
                medians
                and x is not None
                and confidence is not None
                and hasattr(model, "predict_proba")
                and all(name in medians for name in feature_names)
            ):
                pred_class_index = classes.index(pred)
                impacts = []

                for name in feature_names:
                    altered = x.copy()
                    altered.loc[0, name] = float(medians[name])
                    altered_scaled = scaler.transform(altered)
                    altered_probs = model.predict_proba(altered_scaled)[0]
                    altered_prob = float(altered_probs[pred_class_index])

                    impacts.append({
                        "Feature": name,
                        "Original_Value": float(x.loc[0, name]),
                        "Training_Median": float(medians[name]),
                        "Impact": float(confidence - altered_prob),
                    })

                local = pd.DataFrame(impacts)
                local["Abs_Impact"] = local["Impact"].abs()
                local = local.sort_values(
                    "Abs_Impact",
                    ascending=False,
                )

                chart = local.head(10).sort_values("Impact")
                fig, ax = plt.subplots(figsize=(9, 5))
                ax.barh(chart["Feature"], chart["Impact"])
                ax.axvline(0, linestyle="--")
                ax.set_xlabel(
                    "Original class probability − probability after median replacement"
                )
                ax.grid(axis="x", alpha=0.2)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                shown = local.head(15).copy()
                shown["Impact (%)"] = (shown["Impact"] * 100).round(2)
                st.dataframe(
                    shown[
                        [
                            "Feature",
                            "Original_Value",
                            "Training_Median",
                            "Impact (%)",
                        ]
                    ],
                    use_container_width=True,
                )
            else:
                st.info(
                    "模型檔未提供完整 training_medians，或模型不支援機率輸出；"
                    "因此略過 Local Feature Impact。"
                )

            st.subheader("⑧ 模型評估（訓練時保存）")
            if "test_accuracy" in bundle:
                st.metric(
                    "Hold-out Test Accuracy",
                    f"{float(bundle['test_accuracy']) * 100:.2f}%",
                )
                st.caption(
                    "這是訓練時保留測試集的結果，不是目前上傳音訊的正確率。"
                )

            if "confusion_matrix" in bundle:
                cm = np.asarray(bundle["confusion_matrix"])
                if cm.ndim == 2:
                    fig, ax = plt.subplots(figsize=(5, 4))
                    im = ax.imshow(cm)
                    tick_labels = ["Cat", "Dog"]
                    ax.set_xticks(range(cm.shape[1]))
                    ax.set_yticks(range(cm.shape[0]))
                    ax.set_xticklabels(tick_labels[:cm.shape[1]])
                    ax.set_yticklabels(tick_labels[:cm.shape[0]])
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("Actual")
                    ax.set_title("Confusion Matrix")
                    for i in range(cm.shape[0]):
                        for j in range(cm.shape[1]):
                            ax.text(
                                j,
                                i,
                                int(cm[i, j]),
                                ha="center",
                                va="center",
                            )
                    fig.colorbar(im, ax=ax)
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

            if "classification_report" in bundle:
                with st.expander("查看 Classification Report"):
                    report_df = pd.DataFrame(bundle["classification_report"]).T
                    st.dataframe(report_df, use_container_width=True)

        st.subheader("⑨ Engineering Interpretation")
        st.markdown(
            """
```text
Audio Input
    ↓
Audio Decode / Mono / 22.05 kHz
    ↓
300–4000 Hz Band-pass Filter
    ↓
22 DSP Features
    ↓
Cat / Dog Random Forest (classification mode only)
    ↓
Prediction + DSP Plots + Explainability
```

**注意：Feature Importance / Local Feature Impact 是模型解釋，不是物理因果證明。**
"""
        )

    except Exception as exc:
        st.error(f"音訊分析失敗：{exc}")
        st.exception(exc)
