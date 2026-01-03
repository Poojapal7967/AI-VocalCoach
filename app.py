import streamlit as st
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import whisper
import librosa
import librosa.display
import matplotlib.pyplot as plt
import time

# --- UI STYLING ---
st.set_page_config(page_title="AI Vocal Coach Pro", page_icon="🎙️", layout="wide")

st.markdown("""
    <style>
    .stApp { background: #0e1117; color: white; }
    [data-testid="stMetricValue"] { color: #ffffff !important; font-size: 50px !important; font-weight: bold !important; }
    [data-testid="stMetricLabel"] { color: #00f2fe !important; font-size: 20px !important; }
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.08) !important;
        border: 2px solid #1e2130 !important;
        padding: 25px !important;
        border-radius: 15px !important;
    }
    .stButton>button {
        background: linear-gradient(45deg, #ff4b4b, #ff7575) !important;
        color: white !important;
        border-radius: 50px !important;
        padding: 20px !important;
        font-size: 20px !important;
        width: 100% !important;
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
st.write("### *Smart Feedback for English & Hindi Speakers*")

# Settings Area
col_set1, col_set2 = st.columns(2)
with col_set1:
    duration = st.slider("Recording Time (seconds)", 3, 15, 5)
with col_set2:
    language_choice = st.selectbox("Apni Language Chuniye", ["English", "Hindi"])
    lang_code = "en" if language_choice == "English" else "hi"

st.markdown("---")

# Recording Button
if st.button("🔴 START RECORDING"):
    fs = 44100
    with st.spinner(f"✨ Recording in {language_choice}... Boliye!"):
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()
        write('speech.wav', fs, recording)
    
    st.success("✅ Voice Captured!")
    st.audio('speech.wav')

    # AI PROCESSING
    with st.spinner("AI Analysis in progress..."):
        # 1. Transcription (Based on selected language)
        result = model.transcribe("speech.wav", language=lang_code)
        text = result['text']
        
        # 2. Audio Waveform Data
        y, sr = librosa.load("speech.wav")
        
        # 3. Calculations
        words = text.split()
        wpm = int(len(words) / (duration / 60))
        fillers = ["um", "uh", "ah", "like", "hmm", "actually", "matlab", "toh"]
        filler_count = sum(1 for word in words if word.lower().strip(",.") in fillers)
        score = max(10, 100 - (filler_count * 15))

    # --- RESULTS ---
    st.markdown("## 📊 Performance Analytics")
    
    # 1. Visual Waveform
    
    fig, ax = plt.subplots(figsize=(10, 2))
    librosa.display.waveshow(y, sr=sr, ax=ax, color='#00f2fe')
    ax.set_axis_off()
    fig.patch.set_facecolor('#0e1117')
    st.pyplot(fig)

    # 2. Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Confidence Score", f"{score}%")
    c2.metric("Filler Words", filler_count)
    c3.metric("Speech Pace", f"{wpm} WPM")

    with st.expander("📝 View Transcription"):
        st.write(f"**Recognized Text:** {text}")

    # 3. Coaching Tips
    st.subheader("💡 Coaching Insights")
    if score > 80:
        st.success("🌟 Great Job! Your flow is very professional.")
    else:
        st.warning("⚡ Tip: Try to avoid filler words (um, uh, matlab).")