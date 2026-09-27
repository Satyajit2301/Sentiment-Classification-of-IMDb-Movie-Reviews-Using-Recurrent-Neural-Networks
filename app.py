"""
IMDb Movie Review Sentiment Analyzer
-------------------------------------
Streamlit front-end for the RNN / LSTM / Bi-LSTM / GRU sentiment
classification project (CSE 4192 - Machine Learning Projects with Python).

Deployment notes
-----------------
1. Put the trained model file next to this script (or in a `models/` folder).
   By default this looks for `best_model.keras` — the file the notebook
   saves in Block 13. Rename accordingly if you kept a different name.
2. requirements.txt lists everything needed for Streamlit Cloud.
3. Preprocessing here MUST mirror the notebook exactly (same VOCAB_SIZE,
   MAX_LEN, index offsets) or predictions will be meaningless — see the
   constants below.
"""

import os
import re

import numpy as np
import streamlit as st
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ------------------------------------------------------------------
# Constants — must match the training notebook (Block 1)
# ------------------------------------------------------------------
VOCAB_SIZE = 10000
MAX_LEN = 100
INDEX_FROM = 3   # Keras IMDb offset: real word index = rank + INDEX_FROM
START_CHAR = 1
OOV_CHAR = 2

MODEL_PATHS = ["best_model.keras", "models/best_model.keras", "model/best_model.keras"]

EXAMPLE_POSITIVE = (
    "This movie completely blew me away. The performances were heartfelt, "
    "the direction was confident, and the story kept me hooked from the "
    "very first scene until the credits rolled. Easily one of the best "
    "films I have seen this year."
)
EXAMPLE_NEGATIVE = (
    "I really wanted to like this movie but it was a complete letdown. "
    "The plot dragged on forever, the acting felt flat, and the ending "
    "made no sense at all. I would not recommend wasting your time on this."
)

# ------------------------------------------------------------------
# Page setup
# ------------------------------------------------------------------
st.set_page_config(page_title="IMDb Sentiment Analyzer", page_icon="🎬", layout="centered")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }

    /* === DARK THEME === */
    .stApp {
        background: linear-gradient(160deg, #0d0d0d 0%, #1a1a2e 40%, #16213e 100%) !important;
        color: #e0e0e0 !important;
    }

    /* All text elements */
    .stApp h1, .stApp h2, .stApp h3, .stApp p, .stApp span, .stApp label,
    .stApp div, .stApp li { color: #e0e0e0 !important; }

    /* Text area */
    .stTextArea textarea {
        background-color: #1e1e2f !important;
        color: #e0e0e0 !important;
        border: 1px solid #3a3a5c !important;
        border-radius: 12px !important;
    }
    .stTextArea textarea::placeholder { color: #888 !important; }

    /* Buttons */
    div.stButton > button {
        border-radius: 10px; font-weight: 600;
        background-color: #2a2a4a !important;
        color: #e0e0e0 !important;
        border: 1px solid #4a4a6a !important;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        background-color: #3a3a6a !important;
        border-color: #7c5cbf !important;
        transform: translateY(-1px);
    }
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #6a3de8, #a855f7) !important;
        border: none !important; color: white !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #7c4dff, #b86bff) !important;
    }

    /* Expander */
    .streamlit-expanderHeader { color: #c0c0e0 !important; }
    .streamlit-expanderContent { background-color: #1a1a2e !important; }

    /* Caption */
    .stCaption, caption { color: #888 !important; }

    /* Progress bar */
    .stProgress > div > div { background-color: #2a2a4a !important; }

    /* Top header / toolbar bar */
    header[data-testid="stHeader"],
    .stHeader, [data-testid="stHeader"] {
        background-color: #0d0d0d !important;
        border-bottom: 1px solid #1a1a2e !important;
    }
    /* Streamlit toolbar / hamburger menu */
    [data-testid="stToolbar"],
    .stToolbar, #MainMenu {
        background-color: transparent !important;
        color: #e0e0e0 !important;
    }
    [data-testid="stDecoration"] { display: none !important; }

    /* Download button */
    .stDownloadButton > button,
    div.stDownloadButton > button {
        background-color: #2a2a4a !important;
        color: #e0e0e0 !important;
        border: 1px solid #4a4a6a !important;
        border-radius: 10px !important;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stDownloadButton > button:hover,
    div.stDownloadButton > button:hover {
        background-color: #3a3a6a !important;
        border-color: #7c5cbf !important;
    }

    /* Warning / Error / Info boxes */
    .stAlert, [data-testid="stAlert"],
    div[data-testid="stNotification"] {
        background-color: #1e1e2f !important;
        color: #e0e0e0 !important;
        border-color: #3a3a5c !important;
    }

    /* Expander full dark */
    details, summary, [data-testid="stExpander"] {
        background-color: transparent !important;
        color: #c0c0e0 !important;
    }
    [data-testid="stExpanderDetails"] {
        background-color: #12121f !important;
        border-color: #2a2a4a !important;
    }

    /* Any remaining white containers */
    .block-container, [data-testid="stAppViewContainer"],
    [data-testid="stAppViewBlockContainer"],
    .main .block-container, section[data-testid="stSidebar"],
    .stMainBlockContainer {
        background-color: transparent !important;
    }
    .main { background-color: transparent !important; }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: #0d0d0d; }
    ::-webkit-scrollbar-thumb { background: #3a3a5c; border-radius: 4px; }

    /* Hero */
    .hero {
        background: linear-gradient(135deg, #1f1147 0%, #4a2a8c 45%, #a4508b 100%);
        padding: 2rem 1.5rem; border-radius: 18px; text-align: center;
        color: white; margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(106, 61, 232, 0.3);
    }
    .hero h1 { margin: 0; font-size: 2rem; font-weight: 700; color: white !important; }
    .hero p { margin: 0.4rem 0 0 0; opacity: 0.85; font-size: 0.95rem; color: #ddd !important; }

    /* Result card */
    .result-card {
        padding: 1.4rem 1.6rem; border-radius: 16px; margin-top: 1.2rem;
        border-left: 8px solid var(--accent);
        background: var(--bg); animation: fadeIn 0.5s ease;
    }
    .result-card h2 { margin: 0; color: var(--accent) !important; font-size: 1.6rem; }
    .result-card p { margin: 0.3rem 0 0 0; color: #d0d0d0 !important; }

    @keyframes fadeIn { from {opacity:0; transform: translateY(6px);} to {opacity:1; transform: translateY(0);} }
    </style>

    <div class="hero">
        <h1>🎬 IMDb Movie Review Sentiment Analyzer</h1>
        <p>Powered by a Recurrent Neural Network trained on 50,000 IMDb reviews</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Cached resources
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading trained model...")
def get_model():
    for path in MODEL_PATHS:
        if os.path.exists(path):
            return load_model(path), path
    return None, None


@st.cache_resource(show_spinner="Loading IMDb vocabulary...")
def get_word_index():
    raw = imdb.get_word_index()
    return {word: rank + INDEX_FROM for word, rank in raw.items()}


model, model_path = get_model()
word_index = get_word_index()

# ------------------------------------------------------------------
# Preprocessing — mirrors tensorflow.keras.datasets.imdb encoding
# ------------------------------------------------------------------
def clean_and_tokenize(text: str):
    text = text.lower()
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"[^a-z']+", " ", text)
    return text.split()


def encode_review(text: str):
    tokens = clean_and_tokenize(text)
    seq = [START_CHAR]
    for tok in tokens:
        idx = word_index.get(tok, OOV_CHAR)
        seq.append(idx if idx < VOCAB_SIZE else OOV_CHAR)
    return seq


def predict_sentiment(text: str):
    seq = encode_review(text)
    padded = pad_sequences([seq], maxlen=MAX_LEN, padding="post", truncating="post")
    prob = float(model.predict(padded, verbose=0)[0][0])
    label = "Positive" if prob >= 0.5 else "Negative"
    confidence = prob if prob >= 0.5 else 1 - prob
    return label, prob, confidence


# ------------------------------------------------------------------
# UI — input
# ------------------------------------------------------------------
if "review_text" not in st.session_state:
    st.session_state.review_text = ""

ex_col1, ex_col2, ex_col3 = st.columns(3)
if ex_col1.button("💚 Positive example", use_container_width=True):
    st.session_state.review_text = EXAMPLE_POSITIVE
    st.rerun()
if ex_col2.button("💔 Negative example", use_container_width=True):
    st.session_state.review_text = EXAMPLE_NEGATIVE
    st.rerun()
if ex_col3.button("🗑️ Clear", use_container_width=True):
    st.session_state.review_text = ""
    st.rerun()

st.text_area(
    "Write your movie review below, then click Analyze:",
    key="review_text",
    height=180,
    placeholder="e.g. This movie was an absolute masterpiece...",
)

analyze_clicked = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)

# ------------------------------------------------------------------
# UI — result
# ------------------------------------------------------------------
if analyze_clicked:
    text = st.session_state.review_text.strip()
    if not text:
        st.warning("Please write a review before analyzing.")
    elif model is None:
        st.error(
            "Couldn't find a trained model file. Place `best_model.keras` "
            f"in the app folder (checked: {', '.join(MODEL_PATHS)})."
        )
    else:
        label, prob, confidence = predict_sentiment(text)
        is_pos = label == "Positive"
        accent = "#22c55e" if is_pos else "#ef4444"
        bg = "#1a2e1a" if is_pos else "#2e1a1a"
        emoji = "😀" if is_pos else "😞"

        # Soft dark tint reacting to the verdict
        page_bg = (
            "linear-gradient(160deg, #0d1a0d 0%, #1a1a2e 60%)"
            if is_pos
            else "linear-gradient(160deg, #1a0d0d 0%, #1a1a2e 60%)"
        )
        st.markdown(
            f"<style>.stApp {{ background: {page_bg}; transition: background 0.6s ease; }}</style>",
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="result-card" style="--accent:{accent}; --bg:{bg};">
                <h2>{emoji} {label}</h2>
                <p>Confidence: <b>{confidence * 100:.1f}%</b> &nbsp;|&nbsp;
                   Raw model score: <b>{prob:.3f}</b> (0 = very negative, 1 = very positive)</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(prob)

        if is_pos and confidence > 0.9:
            st.balloons()

        result_txt = (
            f"Review:\n{text}\n\n"
            f"Predicted Sentiment: {label}\n"
            f"Confidence: {confidence * 100:.2f}%\n"
            f"Raw model score: {prob:.4f}\n"
        )
        st.download_button(
            "📥 Download this result",
            data=result_txt,
            file_name="sentiment_result.txt",
            mime="text/plain",
        )

# ------------------------------------------------------------------
# About / footer
# ------------------------------------------------------------------
with st.expander("ℹ️ About this model"):
    if model is not None:
        st.write(f"**Architecture:** `{model.name}`")
        st.write(f"**Trainable parameters:** {model.count_params():,}")
        st.write(f"**Loaded from:** `{model_path}`")
    st.write(
        "Trained on the Keras IMDb dataset (50,000 labeled reviews) using "
        "a vocabulary of the top 10,000 words and sequences padded/truncated "
        "to 100 tokens. Four architectures — Simple RNN, LSTM, Bi-LSTM, and "
        "GRU — were compared, and the best model by ROC-AUC was deployed here."
    )

st.caption("CSE 4192 · Machine Learning Projects with Python · Sentiment Classification of IMDb Movie Reviews")
