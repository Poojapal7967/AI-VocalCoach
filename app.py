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
    .stApp {
        background: radial-gradient(circle at center, #1a1a3a 0%, #050510 100%);
        color: #ffffff;
    }
    [data-testid="stMetricValue"] { color: #ffffff !important; font-size: 55px !important; font-weight: 900 !important; }
    [data-testid="stMetricLabel"] { color: #00f2fe !important; font-size: 18px !important; }
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 20px !important;
        padding: 30px !important;
    }
    .stButton>button {
        width: 100% !important;
        background: linear-gradient(90deg, #7000ff, #00f2fe) !important;
        color: white !important;
        border-radius: 50px !important;
        padding: 20px !important;
        font-size: 22px !important;
        border: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return whisper.load_model("base")

model = load_model()

# --- APP UI ---
st.title("🎙️ AI Vocal Coach Pro")
st.write("<p style='text-align: center; font-size: 20px; opacity: 0.9;'>Master Your Voice with Specific Goals</p>", unsafe_allow_html=True)

# Settings Row
st.write("##")
c1, c2, c3 = st.columns(3)

with c1:
    duration = st.select_slider("Session Length (sec)", options=[5, 10, 15], value=5)
with c2:
    language = st.selectbox("Language Mode", ["English", "Hindi"])
with c3:
    # --- NAYA OPTION: PREPARATION GOAL ---
    goal = st.selectbox("What are you preparing for?", 
                        ["Public Speaking", "Anchoring", "Motivational Speaking", "Teaching", "Interview", "Storytelling"])

st.write("##")
if st.button(f"✨ START {goal.upper()} SESSION"):
    fs = 44100
    with st.spinner(f"🎤 Recording for {goal}... Boliye!"):
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
    
    # Custom Logic Based on Goal
    score = 100
    if goal == "Teaching" and wpm > 150: score -= 20 # Teaching should be slow
    if goal == "Anchoring" and wpm < 130: score -= 15 # Anchoring needs energy/speed
    score = max(10, score - (text.lower().count("um") * 10))

    # --- REPORT ---
    st.markdown(f"### 📊 Analysis for: {goal}")
    
    fig, ax = plt.subplots(figsize=(12, 2))
    librosa.display.waveshow(y, sr=sr, ax=ax, color='#7000ff')
    ax.set_axis_off()
    fig.patch.set_facecolor('#050510')
    st.pyplot(fig)

    m1, m2, m3 = st.columns(3)
    m1.metric("Goal Match Score", f"{score}%")
    m2.metric("Speech Rate", f"{wpm} WPM")
    m3.metric("Goal Type", goal)

    st.info(f"**Transcription:** {text}")

    # --- CUSTOM FEEDBACK BASED ON GOAL ---
    st.subheader("💡 Personalized Advice")
    if goal == "Public Speaking":
        st.write("👉 Focus on eye contact and hand gestures. Your speed is good!")
    elif goal == "Teaching":
        if wpm > 140: st.warning("👉 Aap thoda tez bol rahe hain. Students ko samajhne ke liye thoda slow bolein.")
        else: st.success("👉 Perfect pace for teaching!")
    elif goal == "Anchoring":
        st.write("👉 Energy high rakhein aur words par emphasis dein.")