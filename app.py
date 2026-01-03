import streamlit as st
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import whisper
import librosa
import librosa.display
import matplotlib.pyplot as plt
import time

# --- PREMIUM UI CONFIG ---
st.set_page_config(page_title="AI Vocal Coach Pro", page_icon="🎙️", layout="wide")

st.markdown("""
    <style>
    .stApp { background: #0b0e14; color: #ffffff; }
    
    /* Metrics Box - Full Width & Centered */
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 45px !important;
        font-weight: 800 !important;
    }
    
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(0, 242, 254, 0.3);
        padding: 20px;
        border-radius: 20px;
        text-align: center;
    }

    /* Full Width Button */
    .stButton>button {
        width: 100% !important;
        background: linear-gradient(90deg, #00f2fe 0%, #4facfe 100%) !important;
        color: black !important;
        font-weight: bold !important;
        border-radius: 12px !important;
        height: 3em !important;
        font-size: 20px !important;
    }
    
    /* Progress Bar Color */
    .stProgress > div > div > div > div {
        background-color: #00f2fe !important;
    }
    </style>
    """, unsafe_allow_html=True)

# AI Model
@st.cache_resource
def load_model():
    return whisper.load_model("base")

model = load_model()

# --- MAIN UI ---
st.title("🎙️ AI Vocal Coach Pro")
st.write("### *Analyze your voice with professional AI insights*")
st.markdown("---")

# User Friendly Settings
st.subheader("🛠️ Quick Setup")
duration = st.select_slider("Select Recording Duration (Seconds)", options=[3, 5, 10, 15], value=5)
language_choice = st.radio("Choose Language", ["English", "Hindi"], horizontal=True)
lang_code = "en" if language_choice == "English" else "hi"

st.markdown("---")

# Recording Action
if st.button("🎤 CLICK TO START RECORDING"):
    fs = 44100
    with st.spinner("✨ Listening to your voice..."):
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()
        write('speech.wav', fs, recording)
    
    st.success("✅ Voice Captured! Analyzing now...")
    st.audio('speech.wav')

    # AI Brain
    with st.spinner("🤖 AI is thinking..."):
        result = model.transcribe("speech.wav", language=lang_code)
        text = result['text']
        y, sr = librosa.load("speech.wav")
        
        words = text.split()
        wpm = int(len(words) / (duration / 60))
        fillers = ["um", "uh", "ah", "like", "hmm", "matlab", "toh"]
        filler_count = sum(1 for word in words if word.lower().strip(",.") in fillers)
        score = max(5, 100 - (filler_count * 15))

    # --- FULL WIDTH RESULTS ---
    st.markdown("## 📈 Performance Report")
    
    # Waveform Graph
    fig, ax = plt.subplots(figsize=(12, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax, color='#00f2fe', alpha=0.8)
    ax.set_axis_off()
    fig.patch.set_facecolor('#0b0e14')
    st.pyplot(fig)

    # Metrics in a clean row
    m1, m2, m3 = st.columns(3)
    m1.metric("Confidence Score", f"{score}%")
    m2.metric("Filler Words", filler_count)
    m3.metric("Speech Rate (WPM)", wpm)

    st.markdown("---")
    st.subheader("📝 Transcription")
    st.info(text)

    # Stylish Coaching Insights
    with st.container():
        st.subheader("💡 Expert Advice")
        if score < 70:
            st.warning("👉 Aapne kaafi fillers use kiye hain. Bolte waqt chote pauses lene ki koshish karein.")
        else:
            st.success("👉 Great flow! Aapki clarity kaafi achi hai.")