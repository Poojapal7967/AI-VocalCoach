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
    .stApp { background: radial-gradient(circle at center, #1a1a3a 0%, #050510 100%); color: #ffffff; }
    [data-testid="stMetricValue"] { color: #ffffff !important; font-size: 45px !important; font-weight: 900 !important; }
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 20px !important;
    }
    .stButton>button {
        width: 100% !important;
        background: linear-gradient(90deg, #7000ff, #00f2fe) !important;
        color: white !important;
        border-radius: 50px !important;
        padding: 15px !important;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return whisper.load_model("base")

model = load_model()

# --- APP UI ---
st.title("🎙️ AI Vocal Coach Pro")
st.write("<p style='text-align: center;'>Analyze your Pitch, Tone, and Confidence</p>", unsafe_allow_html=True)

# Settings
c1, c2, c3 = st.columns(3)
with c1: duration = st.select_slider("Duration (sec)", options=[5, 10, 15], value=5)
with c2: language = st.selectbox("Language", ["English", "Hindi"])
with c3: goal = st.selectbox("Goal", ["Public Speaking", "Anchoring", "Teaching", "Interview"])

if st.button(f"✨ START {goal.upper()} SESSION"):
    fs = 44100
    with st.spinner("🎤 Recording..."):
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()
        write('speech.wav', fs, recording)
    
    # Analysis Brain
    result = model.transcribe("speech.wav", language=("en" if language=="English" else "hi"))
    text = result['text']
    y, sr = librosa.load("speech.wav")
    
    # --- PITCH & TONE CALCULATION ---
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    avg_pitch = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
    pitch_variation = np.std(pitches[pitches > 0]) if np.any(pitches > 0) else 0
    
    wpm = int(len(text.split()) / (duration / 60))
    
    # Tone Logic
    tone = "Neutral"
    if pitch_variation > 100: tone = "Dynamic & Energetic"
    elif pitch_variation < 30: tone = "Monotone (Flat)"
    
    # --- REPORT ---
    st.markdown("---")
    st.subheader(f"📊 Live Analysis: {goal}")
    
    # Waveform
    fig, ax = plt.subplots(figsize=(12, 2.5))
    librosa.display.waveshow(y, sr=sr, ax=ax, color='#00f2fe')
    ax.set_axis_off()
    fig.patch.set_facecolor('#050510')
    st.pyplot(fig)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Avg Pitch", f"{int(avg_pitch)} Hz")
    m2.metric("Voice Tone", tone)
    m3.metric("Speech Rate", f"{wpm} WPM")
    m4.metric("Confidence", "High" if wpm > 100 else "Low")

    st.info(f"**Transcription:** {text}")
    
    # Coaching Insight based on Tone
    st.subheader("💡 Vocal Coaching Insight")
    if tone == "Monotone (Flat)":
        st.warning("👉 Aapki awaaz thodi flat hai. Bolte waqt emotions aur pitch variation laane ki koshish karein.")
    else:
        st.success("👉 Good Job! Aapki tone kaafi dynamic aur engaging hai.")