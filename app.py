import streamlit as st
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import whisper
import librosa
import librosa.display
import matplotlib.pyplot as plt

# --- FUTURISTIC GLASS UI STYLING ---
st.set_page_config(page_title="AI Vocal Coach Pro", page_icon="🤖", layout="wide")

st.markdown("""
    <style>
    /* Main Background with Purple-Blue Gradient */
    .stApp {
        background: radial-gradient(circle at center, #1e1e3f 0%, #0b0b1a 100%);
        color: #ffffff;
    }
    
    /* Frosted Glass Cards */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(15px) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 25px !important;
        padding: 25px !important;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37) !important;
    }

    /* Glow Effect for Titles */
    h1 {
        text-shadow: 0 0 20px rgba(173, 0, 255, 0.5);
        font-family: 'Inter', sans-serif;
        text-align: center;
    }

    /* Neon Button */
    .stButton>button {
        width: 100% !important;
        background: linear-gradient(90deg, #ad00ff, #4facfe) !important;
        color: white !important;
        font-weight: bold !important;
        border-radius: 30px !important;
        border: none !important;
        height: 3.5em !important;
        transition: 0.3s all ease;
    }
    
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 25px rgba(173, 0, 255, 0.6);
    }

    /* Waveform Container */
    .plot-container {
        border-radius: 20px;
        overflow: hidden;
    }
    </style>
    """, unsafe_allow_html=True)

# Model Load
@st.cache_resource
def load_model():
    return whisper.load_model("base")

model = load_model()

# --- APP LAYOUT ---
st.title("🤖 Make It Awesome")
st.write("<p style='text-align: center; opacity: 0.8;'>Your AI Mentor for Perfect Speech</p>", unsafe_allow_html=True)

st.markdown("---")

# Settings in Glass Cards
col_set1, col_set2 = st.columns(2)
with col_set1:
    duration = st.select_slider("Session Length", options=[5, 10, 15], value=5)
with col_set2:
    language = st.selectbox("Language Mode", ["English", "Hindi"])

# Center Button
st.write("##")
if st.button("✨ START YOUR SESSION"):
    fs = 44100
    with st.spinner("🤖 AI is listening..."):
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()
        write('speech.wav', fs, recording)
    
    st.audio('speech.wav')

    # Analysis
    result = model.transcribe("speech.wav", language=("en" if language=="English" else "hi"))
    text = result['text']
    y, sr = librosa.load("speech.wav")
    
    words = text.split()
    wpm = int(len(words) / (duration / 60))
    score = max(10, 100 - (text.lower().count("um") * 10))

    # --- SHOW RESULTS ---
    st.markdown("### 📊 Performance Analytics")
    
    # Waveform with Glow
    fig, ax = plt.subplots(figsize=(12, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax, color='#ad00ff', alpha=0.7)
    ax.set_axis_off()
    fig.patch.set_facecolor('#0b0b1a')
    st.pyplot(fig)

    res_col1, res_col2, res_col3 = st.columns(3)
    res_col1.metric("Confidence", f"{score}%")
    res_col2.metric("Speech Rate", f"{wpm} WPM")
    res_col3.metric("Language", language)

    st.info(f"**Transcription:** {text}")