import streamlit as st
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import whisper
import librosa
import time

# --- ULTIMATE HIGH-CONTRAST UI STYLING ---
st.set_page_config(page_title="AI Vocal Coach Pro", page_icon="🎙️", layout="wide")

st.markdown("""
    <style>
    /* Dark Premium Background */
    .stApp {
        background: #0e1117;
        color: white;
    }
    
    /* Metrics Styling - Pure White Numbers */
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 50px !important;
        font-weight: bold !important;
    }
    
    /* Metric Labels */
    [data-testid="stMetricLabel"] {
        color: #00f2fe !important;
        font-size: 20px !important;
    }

    /* Card Boxes Styling */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.08) !important;
        border: 2px solid #1e2130 !important;
        padding: 30px !important;
        border-radius: 15px !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.5) !important;
    }

    /* HUGE RED BUTTON STYLING */
    .stButton>button {
        background: linear-gradient(45deg, #ff4b4b, #ff7575) !important;
        color: white !important;
        border-radius: 50px !important;
        padding: 20px !important;
        font-size: 24px !important;
        font-weight: bold !important;
        width: 100% !important;
        margin-top: 20px !important;
        border: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

# AI Model Loading
@st.cache_resource
def load_model():
    return whisper.load_model("base")

model = load_model()

# --- APP UI ---
st.title("🎙️ AI Vocal Coach Pro")
st.write("### Analyze your confidence, tone, and speaking speed.")
st.markdown("---")

# Recording Duration Settings
duration = st.slider("Set Recording Time (seconds)", 3, 15, 5)

# --- MAIN START BUTTON ---
# Isse humne columns se bahar rakha hai taaki ye hamesha dikhe
if st.button("🔴 START RECORDING & ANALYSIS"):
    fs = 44100
    with st.spinner("✨ Recording... Speak Now!"):
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()
        write('speech.wav', fs, recording)
    
    st.success("✅ Voice Captured! Analyzing...")
    st.audio('speech.wav')

    # AI PROCESSING
    with st.spinner("AI is evaluating your performance..."):
        # 1. Transcription
        result = model.transcribe("speech.wav")
        text = result['text']
        
        # 2. Audio Analysis
        y, sr = librosa.load("speech.wav")
        pitches, _ = librosa.piptrack(y=y, sr=sr)
        avg_pitch = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
        
        # 3. Calculation Logic
        words = text.split()
        wpm = int(len(words) / (duration / 60))
        fillers = ["um", "uh", "ah", "like", "hmm"]
        filler_count = sum(1 for word in words if word.lower().strip(",.") in fillers)
        
        score = max(10, 100 - (filler_count * 15))

    # --- FINAL RESULTS DISPLAY ---
    st.markdown("## 📊 Performance Analytics")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Confidence Score", f"{score}%")
    c2.metric("Filler Words", filler_count)
    c3.metric("Speech Pace", f"{wpm} WPM")

    with st.expander("📝 View Transcription"):
        st.write(f"**You said:** *\"{text}\"*")

    # --- TIPS ---
    st.subheader("💡 AI Coaching Insights")
    if score > 80:
        st.success("🌟 Excellent! Your delivery is sharp.")
    else:
        st.warning("⚡ Try to reduce filler words.")
        
    if wpm < 110:
        st.info("🐢 You are speaking a bit slowly.")
    elif wpm > 170:
        st.info("🚀 Speaking too fast!")