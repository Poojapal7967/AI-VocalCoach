import streamlit as st
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import whisper
import librosa
import librosa.display
import matplotlib.pyplot as plt

# --- ULTIMATE NEON GLASS UI ---
st.set_page_config(page_title="AI Vocal Coach Pro", page_icon="🎙️", layout="wide")

st.markdown("""
    <style>
    /* Deep Space Background */
    .stApp {
        background: radial-gradient(circle at center, #1a1a3a 0%, #050510 100%);
        color: #ffffff;
    }
    
    /* Metrics: High Contrast White Numbers */
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 55px !important;
        font-weight: 900 !important;
        text-shadow: 0 0 15px rgba(255, 255, 255, 0.4);
    }
    
    /* Metric Labels: Cyan Glow */
    [data-testid="stMetricLabel"] {
        color: #00f2fe !important;
        font-size: 18px !important;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    /* Glass Cards: No more faded look */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 20px !important;
        padding: 30px !important;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5) !important;
    }

    /* Professional Start Session Button */
    .stButton>button {
        width: 100% !important;
        background: linear-gradient(90deg, #7000ff, #00f2fe) !important;
        color: white !important;
        border-radius: 50px !important;
        padding: 20px !important;
        font-size: 22px !important;
        font-weight: bold !important;
        border: none !important;
        box-shadow: 0 0 20px rgba(112, 0, 255, 0.4) !important;
    }

    /* Transcription Box */
    .stInfo {
        background: rgba(0, 242, 254, 0.1) !important;
        border: 1px solid #00f2fe !important;
        color: #ffffff !important;
        font-size: 18px !important;
        border-radius: 15px !important;
    }
    </style>
    """, unsafe_allow_html=True)

# AI Model
@st.cache_resource
def load_model():
    return whisper.load_model("base")

model = load_model()

# --- APP UI ---
st.title("🎙️ Make It Awesome")
st.write("<p style='text-align: center; font-size: 20px; opacity: 0.9;'>Master Your Voice with AI Coach Pro</p>", unsafe_allow_html=True)

# Settings Row
st.write("##")
c1, c2 = st.columns(2)
with c1:
    duration = st.select_slider("Session Length (sec)", options=[5, 10, 15], value=5)
with c2:
    language = st.selectbox("Language Mode", ["English", "Hindi"])

st.write("##")
if st.button("✨ START YOUR SESSION"):
    fs = 44100
    with st.spinner("✨ Listening... Boliye!"):
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()
        write('speech.wav', fs, recording)
    
    st.audio('speech.wav')

    # Analysis Brain
    result = model.transcribe("speech.wav", language=("en" if language=="English" else "hi"))
    text = result['text']
    y, sr = librosa.load("speech.wav")
    
    words = text.split()
    wpm = int(len(words) / (duration / 60))
    score = max(10, 100 - (text.lower().count("um") * 15))

    # --- PERFORMANCE SECTION ---
    st.markdown("---")
    st.subheader("📊 Live Analysis Report")
    
    # Waveform (Purple Glow)
    fig, ax = plt.subplots(figsize=(12, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax, color='#7000ff', alpha=0.9)
    ax.set_axis_off()
    fig.patch.set_facecolor('#050510')
    st.pyplot(fig)

    # Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Confidence", f"{score}%")
    m2.metric("Speed (WPM)", wpm)
    m3.metric("Lang Mode", language)

    st.write("##")
    st.info(f"**AI Transcription:** {text}")