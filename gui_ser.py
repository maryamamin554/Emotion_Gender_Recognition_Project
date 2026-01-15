import os
import io
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import joblib
import sounddevice as sd
from scipy.cluster.hierarchy import dendrogram

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

import streamlit as st

# =========================
# CONFIG & PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "ser_model_best.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "ser_scaler.joblib")
GENDER_MODEL_PATH = os.path.join(BASE_DIR, "ser_model_gender.joblib")
RESULTS_CSV_PATH = os.path.join(BASE_DIR, "model_results.csv")
EVAL_DATA_PATH = os.path.join(BASE_DIR, "eval_data_emotion.npz")

DBSCAN_VIS_PATH = os.path.join(BASE_DIR, "dbscan_vis.npz")
HIER_VIS_PATH   = os.path.join(BASE_DIR, "hierarchical_vis.npz")

st.set_page_config(
    page_title="Speech Emotion & Gender – Classical ML",
    layout="wide",
)

sns.set_theme(style="whitegrid", palette="viridis")


# =========================
# LOAD ARTIFACTS
# =========================
@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Emotion model not found at: {MODEL_PATH}")
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler not found at: {SCALER_PATH}")

    model_ = joblib.load(MODEL_PATH)
    scaler_ = joblib.load(SCALER_PATH)

    # optional gender model
    gender_model_ = None
    if os.path.exists(GENDER_MODEL_PATH):
        gender_model_ = joblib.load(GENDER_MODEL_PATH)

    # training metrics
    results_df_ = None
    if os.path.exists(RESULTS_CSV_PATH):
        results_df_ = pd.read_csv(RESULTS_CSV_PATH)

    # evaluation data for confusion matrix
    eval_data_ = None
    if os.path.exists(EVAL_DATA_PATH):
        npz = np.load(EVAL_DATA_PATH, allow_pickle=True)
        eval_data_ = {
            "X_test_s": npz["X_test_s"],
            "y_test_em": npz["y_test_em"],
            "labels": npz["labels"],
        }

    # DBSCAN visualization data
    dbscan_vis_ = None
    if os.path.exists(DBSCAN_VIS_PATH):
        npz_db = np.load(DBSCAN_VIS_PATH, allow_pickle=True)
        dbscan_vis_ = {
            "X_pca": npz_db["X_pca"],
            "clusters": npz_db["clusters"],
            "y_sub_em": npz_db["y_sub_em"],
        }

    # Hierarchical / dendrogram data
    hier_vis_ = None
    if os.path.exists(HIER_VIS_PATH):
        npz_h = np.load(HIER_VIS_PATH, allow_pickle=True)
        hier_vis_ = {
            "Z": npz_h["Z"],
            "labels_small": npz_h["labels_small"],
        }

    return model_, scaler_, gender_model_, results_df_, eval_data_, dbscan_vis_, hier_vis_

try:
    model, scaler, gender_model, results_df, eval_data, dbscan_vis, hier_vis = load_artifacts()
    MODEL_OK = True
except Exception as e:
    MODEL_OK = False
    st.error(f"Error loading model/scaler: {e}")


# =========================
# FEATURE EXTRACTION
# (same as training)
# =========================
def extract_features_from_array(y, sr=22050, n_mfcc=20):
    # pad or repeat to 1 second
    if len(y) < sr:
        y = np.tile(y, int(np.ceil(sr / len(y))))[:sr]

    # simple normalisation (same as training script)
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfccs_delta = librosa.feature.delta(mfccs)
    mfccs_delta2 = librosa.feature.delta(mfccs, order=2)

    feat = np.concatenate([
        mfccs.mean(axis=1), mfccs.std(axis=1),
        mfccs_delta.mean(axis=1), mfccs_delta.std(axis=1),
        mfccs_delta2.mean(axis=1), mfccs_delta2.std(axis=1),
    ])
    return feat


def extract_features_from_bytes(wav_bytes, target_sr=22050):
    with sf.SoundFile(io.BytesIO(wav_bytes)) as s:
        y = s.read(dtype="float32")
        sr = s.samplerate

    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

    feat = extract_features_from_array(y, sr=target_sr)
    return feat, y, target_sr


# =========================
# SIMPLE GENDER ESTIMATION (PITCH-BASED)
# =========================
def estimate_gender_from_signal(y, sr):
    """
    Estimate gender from fundamental frequency (pitch).
    This is just for extra info; the 'main' gender prediction
    can come from the trained classical ML model if available.
    """
    try:
        # Trim silence
        y_trimmed, _ = librosa.effects.trim(y, top_db=30)

        f0, voiced_flag, voiced_prob = librosa.pyin(
            y_trimmed,
            fmin=50,
            fmax=300,
            sr=sr
        )
        if f0 is None:
            return "unknown", 0.0

        mask = (voiced_flag == True) & (~np.isnan(f0))
        if voiced_prob is not None:
            mask = mask & (voiced_prob > 0.9)

        f0_voiced = f0[mask]
        if len(f0_voiced) == 0:
            return "unknown", 0.0

        median_f0 = float(np.median(f0_voiced))
        threshold_hz = 155.0
        gender = "male" if median_f0 < threshold_hz else "female"

        return gender, median_f0
    except Exception:
        return "unknown", 0.0


# =========================
# MIC RECORDING HELPERS
# =========================
def record_audio(duration_sec=5, fs=22050):
    samples = int(duration_sec * fs)
    audio = sd.rec(samples, samplerate=fs, channels=1, dtype="float32")
    sd.wait()
    return audio.flatten(), fs


def array_to_wav_bytes(y, sr):
    buffer = io.BytesIO()
    sf.write(buffer, y, sr, format="WAV")
    buffer.seek(0)
    return buffer.read()


# =========================
# SESSION STATE
# =========================
if "mode1_history" not in st.session_state:
    # list of dicts: {"emotion": ..., "gender_model": ..., "gender_pitch": ...}
    st.session_state.mode1_history = []

if "mode2_history" not in st.session_state:
    # list of dicts: time, name, gender_model, gender_pitch, median_f0, emotion
    st.session_state.mode2_history = []


# =========================
# SIDEBAR NAVIGATION
# =========================
st.sidebar.title("Voice GUI – Classical ML")

mode = st.sidebar.radio(
    "Select option:",
    (
        "1️⃣ Upload Audio (Emotion / Gender)",
        "2️⃣ Your Voice (Record – Gender + Emotion)",
        "3️⃣ Training Visualizations",
    ),
)

if st.sidebar.button("Clear session history"):
    st.session_state.mode1_history = []
    st.session_state.mode2_history = []
    st.sidebar.success("Cleared history for both modes.")


# =========================
# MODE 1: UPLOAD AUDIO
# =========================
# =========================
# MODE 1: UPLOAD AUDIO
# (Clean: NO pitch-based gender)
# + Adds a gender scatter plot (over the session timeline)
# =========================
if mode.startswith("1️⃣"):
    st.title("🎧 Upload Audio – Emotion")

    st.write(
        """
        **Mode 1 – Test audio files**

        - Upload a `.wav` file from your computer.  
        - The **classical ML model** predicts **emotion**.  
        - If a **gender model** is available, it predicts **gender** too.  
        - Below, you’ll see **session plots** for emotions and gender predictions so far.
        """
    )

    left_col, right_col = st.columns([1.1, 1])

    with left_col:
        uploaded_file = st.file_uploader(
            "Upload a .wav file",
            type=["wav"],
            accept_multiple_files=False,
        )

        if not MODEL_OK:
            st.error("Model or scaler could not be loaded. Fix that first.")
        elif uploaded_file is not None:
            if st.button("🔍 Predict from File"):
                wav_bytes = uploaded_file.read()
                try:
                    feat, y_sig, sr = extract_features_from_bytes(wav_bytes)
                    X = scaler.transform([feat])

                    # Emotion prediction
                    emotion = model.predict(X)[0]
                    prob = None
                    if hasattr(model, "predict_proba"):
                        prob = float(model.predict_proba(X).max())

                    # Gender via ML model (if available)
                    gender_ml = None
                    if gender_model is not None:
                        gender_ml = gender_model.predict(X)[0]

                    st.session_state.mode1_history.append(
                        {
                            "emotion": emotion,
                            "gender_model": gender_ml or "n/a",
                        }
                    )

                    st.audio(wav_bytes, format="audio/wav")

                    st.success(f"Predicted emotion: **{emotion}**")
                    if prob is not None:
                        st.info(f"Model confidence: **{prob:.3f}**")

                    if gender_ml is not None:
                        st.markdown(f"- Gender (ML model): **{str(gender_ml).upper()}**")
                    else:
                        st.markdown("- Gender (ML model): **N/A (gender model not loaded)**")

                except Exception as e:
                    st.error(f"Prediction error: {e}")

    with right_col:
        st.subheader("Session distributions")

        if len(st.session_state.mode1_history) == 0:
            st.write("_No samples yet. Upload a file and click Predict._")
        else:
            df_hist = pd.DataFrame(st.session_state.mode1_history)

            # Emotion distribution
            st.markdown("**Emotion distribution**")
            emo_counts = df_hist["emotion"].value_counts()
            st.bar_chart(emo_counts, height=250)

            # Gender distribution (ML model)
            st.markdown("**Gender distribution (ML model)**")
            gender_ml_counts = df_hist["gender_model"].value_counts()
            st.bar_chart(gender_ml_counts, height=200)

            # Gender scatter plot (ML model) over time
            st.markdown("**Gender scatter (ML model over session order)**")

            # Map labels to numeric for scatter
            # (Adjust labels here if your model outputs different strings)
            label_map = {
                "male": 1,
                "m": 1,
                "1": 1,
                "female": 0,
                "f": 0,
                "0": 0,
                "n/a": None,
            }

            # Build x (sample index) and y (encoded gender)
            x_vals, y_vals = [], []
            for i, g in enumerate(df_hist["gender_model"].astype(str).str.lower().tolist(), start=1):
                y = label_map.get(g, None)
                if y is not None:
                    x_vals.append(i)
                    y_vals.append(y)

            if len(x_vals) == 0:
                st.info("No valid ML gender predictions yet (or gender model not loaded).")
            else:
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots()
                ax.scatter(x_vals, y_vals)
                ax.set_xlabel("Sample #")
                ax.set_ylabel("Gender (0 = Female, 1 = Male)")
                ax.set_yticks([0, 1])
                ax.set_yticklabels(["Female", "Male"])
                ax.set_title("Gender predictions over time (ML model)")
                st.pyplot(fig)



# =========================
# MODE 2: RECORD VOICE
# =========================
elif mode.startswith("2️⃣"):
    st.title("🎙️ Your Voice – Gender + Emotion (Runtime Recording)")

    st.write(
        """
        **Mode 2 – Use your own voice (microphone)**

        - Enter your **name**.  
        - Click the button to **record ~10 seconds** from your microphone.  
        - We predict **emotion** with the trained classical ML model.  
        - Gender:
          - via **gender ML model** (if available), and  
          - via **pitch analysis** (median fundamental frequency).  
        - Below you’ll see plots for **gender** and **emotion** over this session.
        """
    )

    if not MODEL_OK:
        st.error("Model or scaler could not be loaded. Fix that first.")
    else:
        name = st.text_input("Your name")

        if st.button("🎙️ Record 5 seconds & Predict"):
            try:
                st.info("Recording... please speak clearly into your microphone.")
                y_sig, sr = record_audio(duration_sec=5, fs=22050)
                st.success("Recording complete. Cleaning signal & running prediction...")

                # Trim silence and normalise (for both pitch & features)
                y_sig, _ = librosa.effects.trim(y_sig, top_db=30)
                if np.max(np.abs(y_sig)) > 0:
                    y_sig = y_sig / np.max(np.abs(y_sig))

                # Emotion prediction
                feat = extract_features_from_array(y_sig, sr=sr)
                X = scaler.transform([feat])
                emotion = model.predict(X)[0]
                prob = None
                if hasattr(model, "predict_proba"):
                    prob = float(model.predict_proba(X).max())

                # Gender via ML model (if available)
                gender_ml = None
                if gender_model is not None:
                    gender_ml = gender_model.predict(X)[0]

                # Gender estimation using pitch
                gender_pitch, median_f0 = estimate_gender_from_signal(y_sig, sr)

                record = {
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "name": name.strip() or "(no name)",
                    "gender_model": gender_ml or "n/a",
                    "gender_pitch": gender_pitch,
                    "median_f0": median_f0,
                    "emotion": emotion,
                }
                st.session_state.mode2_history.append(record)

                wav_bytes = array_to_wav_bytes(y_sig, sr)
                st.audio(wav_bytes, format="audio/wav")

                summary_lines = [
                    f"Name: **{record['name']}**",
                    f"Emotion: **{emotion}**" + (f" (confidence ≈ {prob:.3f})" if prob is not None else ""),
                ]
                if gender_ml is not None:
                    summary_lines.append(f"Gender (ML model): **{gender_ml.upper()}**")
                summary_lines.append(
                    f"Gender (pitch-based): **{gender_pitch.upper()}** "
                    f"(median F0 ≈ {median_f0:.1f} Hz)"
                )

                st.success("  \n".join(summary_lines))

            except Exception as e:
                st.error(f"Prediction error: {e}")

        st.markdown("---")
        st.subheader("Session plots")

        if len(st.session_state.mode2_history) == 0:
            st.write("_No recordings yet. Click the record button first._")
        else:
            df_hist = pd.DataFrame(st.session_state.mode2_history)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Gender distribution (ML model)**")
                gender_ml_counts = df_hist["gender_model"].value_counts()
                st.bar_chart(gender_ml_counts, height=260)

            with col2:
                st.markdown("**Gender distribution (pitch-based)**")
                gender_p_counts = df_hist["gender_pitch"].value_counts()
                st.bar_chart(gender_p_counts, height=260)

            st.markdown("**Emotion distribution**")
            emotion_counts = df_hist["emotion"].value_counts()
            st.bar_chart(emotion_counts, height=260)

            st.markdown("**Recorded samples (this session)**")
            st.dataframe(df_hist, use_container_width=True)


# =========================
# MODE 3: TRAINING VISUALIZATIONS
# =========================
else:
    st.title("📊 Training Visualizations – Classical ML Models")

    st.write(
        """
        This section shows **offline training results** that were computed in the Jupyter notebook:

        - Model comparison (accuracy & macro F1 from `model_results.csv`)  
        - Confusion matrix of the **best emotion model**  
        - **DBSCAN clustering** in PCA 2D  
        - **Hierarchical clustering dendrogram** on a small subset  
        """
    )

    # ---- Model comparison ----
    st.subheader("1️⃣ Model comparison (from training)")

    if results_df is None:
        st.info("No `model_results.csv` found. Make sure you ran the training notebook and saved metrics.")
    else:
        st.write("Raw metrics table:")
        st.dataframe(results_df, use_container_width=True)

        st.markdown("**Accuracy by model**")
        acc_df = results_df.set_index("model")["accuracy"]
        st.bar_chart(acc_df, height=280)

        st.markdown("**Macro F1 by model**")
        f1_df = results_df.set_index("model")["f1_macro"]
        st.bar_chart(f1_df, height=280)

    st.markdown("---")

    # ---- Confusion matrix ----
    st.subheader("2️⃣ Confusion matrix for best emotion model")

    if (not MODEL_OK) or (eval_data is None):
        st.info(
            "Evaluation data (`eval_data_emotion.npz`) or model not found. "
            "Make sure you saved it from the training notebook."
        )
    else:
        X_test_s = eval_data["X_test_s"]
        y_test_em = eval_data["y_test_em"]
        labels = list(eval_data["labels"])

        y_pred = model.predict(X_test_s)
        cm = confusion_matrix(y_test_em, y_pred, labels=labels)

        fig_cm, ax_cm = plt.subplots(figsize=(7, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax_cm
        )
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("True")
        ax_cm.set_title("Confusion Matrix – Best Emotion Model")
        st.pyplot(fig_cm)

    st.markdown("---")

    # ---- DBSCAN clustering (PCA 2D) ----
    st.subheader("3️⃣ DBSCAN clustering in PCA space")

    if dbscan_vis is None:
        st.info(
            "No DBSCAN visualization data found (`dbscan_vis.npz`). "
            "Run the DBSCAN cell in your training notebook after adding the save code."
        )
    else:
        X_pca = dbscan_vis["X_pca"]
        clusters = dbscan_vis["clusters"]
        y_sub_em = dbscan_vis["y_sub_em"]

        # Scatter plot: PCA 2D colored by cluster
        fig_scatter, ax_scatter = plt.subplots(figsize=(7, 5))
        scatter = ax_scatter.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=clusters,
            s=20,
        )
        ax_scatter.set_xlabel("PCA 1")
        ax_scatter.set_ylabel("PCA 2")
        ax_scatter.set_title("DBSCAN clustering (PCA 2D view)")
        st.pyplot(fig_scatter)

        # Countplot: cluster vs true emotion
        df_db = pd.DataFrame(
            {"cluster": clusters, "emotion": y_sub_em}
        )
        fig_cnt, ax_cnt = plt.subplots(figsize=(7, 4))
        sns.countplot(
            data=df_db,
            x="cluster",
            hue="emotion",
            ax=ax_cnt,
        )
        ax_cnt.set_title("Cluster membership vs true emotions")
        st.pyplot(fig_cnt)

    st.markdown("---")

    # ---- Hierarchical clustering dendrogram ----
    st.subheader("4️⃣ Hierarchical clustering dendrogram")

    if hier_vis is None:
        st.info(
            "No hierarchical visualization data found (`hierarchical_vis.npz`). "
            "Run the dendrogram cell in your training notebook after adding the save code."
        )
    else:
        Z = hier_vis["Z"]
        labels_small = hier_vis["labels_small"]

        fig_dend, ax_dend = plt.subplots(figsize=(10, 4))
        dendrogram(
            Z,
            labels=labels_small,
            leaf_rotation=90,
            leaf_font_size=8,
            color_threshold=0.7 * np.max(Z[:, 2]),
            ax=ax_dend,
        )
        ax_dend.set_title("Hierarchical Clustering Dendrogram (Ward)")
        ax_dend.set_ylabel("Distance")
        st.pyplot(fig_dend)
