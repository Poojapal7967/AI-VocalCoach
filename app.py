import streamlit as st
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import whisper
import librosa
import time

# --- ADVANCED UI STYLING (Neon White Metrics) ---
st.set_page_config(page_title="AI Vocal Coach Pro", page_icon="🎙️", layout="wide")

st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background: radial-gradient(circle at top right, #1e2130, #0e1117);
        color: #e0e0e0;
    }
    
    /* Neon Cards with White Numbers */
    div[data-testid="stMetricValue"] {
        color: #ffffff !important; /* Numbers in Pure White */
        font-size: 40px !important;
        font-weight: 800 !important;
        text-shadow: 0 0 10px rgba(255,255,255,0.3);
    }
    
    div[data-testid="stMetricLabel"] {
        color: #00f2fe !important; /* Labels in Cyan for contrast */
        font-size: 18px !important;
        font-weight: 500 !important;
    }

    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        padding: 25px;
        border-radius: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        border-bottom: 3px solid #00f2fe;
    }

    /* Button Styling */
    .stButton>button {
        width: 100%;
        border-radius: 50px;
        background: linear-gradient(45deg, #00f2fe, #4facfe);
        color: white;
        font-weight: bold;
        border: none;
        padding: 15px;
        box-shadow: 0 4px 15px rgba(79, 172, 254, 0.4);
    }

    /* Tips Box */
    .tips-container {
        background: rgba(255, 255, 255, 0.03);
        padding: 25px;
        border-radius: 15px;
        border-left: 5px solid #00f2fe;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- AI LOGIC ---
@st.cache_resource
def load_model():
    return whisper.load_model("base")

model = load_model()

# --- HEADER ---
st.title("🎙️ AI Vocal Coach Pro")
st.write("### *Analyze your confidence, tone, and speaking speed.*")
st.markdown("---")

col_main, col_sidebar = st.columns([2, 1])

with col_sidebar:
    st.write("### ⚙️ Settings")
    duration = st.slider("Recording Duration (sec)", 3, 15, 5)
    st.info("Higher duration allows for better analysis.")

with col_main:
    if st.button("🔴 Start Live Analysis"):
        fs = 44100
        with st.spinner("✨ Listening..."):
            recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
            sd.wait()
            write('speech.wav', fs, recording)
        
        st.success("✅ Voice Captured!")
        st.audio('speech.wav')

        # AI PROCESSING
        with st.spinner("AI is evaluating your performance..."):
            result = model.transcribe("speech.wav")
            text = result['text']
            y, sr = librosa.load("speech.wav")
            pitches, _ = librosa.piptrack(y=y, sr=sr)
            avg_pitch = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
            
            words = text.split()
            wpm = int(len(words) / (duration / 60))
            fillers = ["um", "uh", "ah", "like", "hmm"]
            filler_count = sum(1 for word in words if word.lower().strip(",.") in fillers)
            
            score = max(5, 100 - (filler_count * 12))

        # --- REPORT UI ---